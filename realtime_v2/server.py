from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Callable
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import librosa
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from realtime_v2.engine import RealtimeRVCConfig, RealtimeRVCEngine
from realtime_v2.model_adapter import RVCModelConfig, RVCRealtimeAdapter


class SileroVAD:
    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        self.threshold = threshold
        self.device = torch.device(device)
        self._model = None
        self._get_speech_timestamps = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._model = model.to(self.device)
        # utils: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
        self._get_speech_timestamps = utils[0]

    def reset(self):
        if self._model is not None:
            self._model.reset_states()

    def warmup(self):
        self._ensure_loaded()
        dummy = np.zeros(self._FRAME_SIZE, dtype=np.float32)
        self.is_speech(dummy)
        self.reset()

    _FRAME_SIZE = 512

    def is_speech(self, samples_16k: np.ndarray) -> tuple[bool, float]:
        self._ensure_loaded()
        audio = samples_16k.astype(np.float32)
        remainder = len(audio) % self._FRAME_SIZE
        if remainder != 0:
            audio = np.pad(audio, (0, self._FRAME_SIZE - remainder))
        max_prob = 0.0
        with torch.no_grad():
            for i in range(0, len(audio), self._FRAME_SIZE):
                frame = torch.from_numpy(audio[i : i + self._FRAME_SIZE]).to(
                    self.device
                )
                p = float(self._model(frame, 16000).item())
                if p > max_prob:
                    max_prob = p
        return max_prob >= self.threshold, max_prob


class SessionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FLUSHING = "flushing"
    STOPPED = "stopped"


class TransportStatus(str, Enum):
    DETACHED = "detached"
    ATTACHED = "attached"


class StartRequest(BaseModel):
    pth_path: str
    index_path: str = ""
    index_rate: float = 0.0
    pitch: int = 0
    formant: float = 0.0
    block_time: float = 0.25
    crossfade_time: float = 0.05
    extra_time: float = 2.5
    input_sr: int = 0
    use_phase_vocoder: bool = False
    input_queue_size: int = 8
    input_ttl_ms: float = 500.0
    output_ttl_ms: float = 500.0
    reconnect_ttl_ms: float = 30000.0
    enable_vad: bool = False
    vad_threshold: float = 0.5


class RttProbeRequest(BaseModel):
    client_ts: float
    seq: int = 0


@dataclass
class InputChunk:
    sequence_id: int
    input_sr: int
    master_samples: np.ndarray
    sample_views: dict[int, np.ndarray]
    features: dict[str, Any]
    created_at: float
    expire_at: float

    def get_view(
        self,
        sr: int,
        *,
        resampler: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> np.ndarray:
        if sr not in self.sample_views:
            if sr == self.input_sr:
                self.sample_views[sr] = self.master_samples
            else:
                self.sample_views[sr] = resampler(self.master_samples, self.input_sr, sr)
        return self.sample_views[sr]

    def set_feature(self, name: str, value: Any):
        self.features[name] = value

    def get_feature(self, name: str, default: Any = None):
        return self.features.get(name, default)


@dataclass
class OutputSlot:
    version: int = 0
    sequence_id: int = -1
    payload: bytes = b""
    sample_count: int = 0
    created_at: float = 0.0
    expire_at: float = 0.0
    infer_time_ms: float = 0.0
    vad_prob: float | None = None


@dataclass
class SessionStats:
    dropped_input_chunks: int = 0
    dropped_output_chunks: int = 0
    expired_input_chunks: int = 0
    expired_output_chunks: int = 0
    accepted_input_chunks: int = 0
    produced_output_chunks: int = 0
    last_infer_time_ms: float = 0.0
    last_output_sequence_id: int = -1
    congested: bool = False
    congestion_since: float | None = None


@dataclass
class AnalysisContext:
    resampler: Callable[[np.ndarray, int, int], np.ndarray]


@dataclass
class ExecutionContext:
    engine: RealtimeRVCEngine
    resampler: Callable[[np.ndarray, int, int], np.ndarray]


@dataclass
class GateDecision:
    action: str
    engine_input: np.ndarray
    block_16k: np.ndarray | None
    output_override: np.ndarray | None = None


class FrameAnalyzer:
    async def analyze(self, chunk: InputChunk, context: AnalysisContext):
        raise NotImplementedError


class FramePolicy:
    def decide(self, chunk: InputChunk, context: ExecutionContext) -> GateDecision:
        raise NotImplementedError


class FrameExecutor:
    async def execute(self, chunk: InputChunk, decision: GateDecision, context: ExecutionContext):
        result = await asyncio.to_thread(
            context.engine.process_block,
            decision.engine_input,
            silence_16k=decision.block_16k,
        )
        output_wave = decision.output_override
        if output_wave is None:
            output_wave = result.output_wave[: chunk.master_samples.shape[0]]
        return output_wave, result.infer_time_ms


class SileroVADAnalyzer(FrameAnalyzer):
    def __init__(self, vad: SileroVAD):
        self.vad = vad

    async def analyze(self, chunk: InputChunk, context: AnalysisContext):
        chunk_16k = chunk.get_view(16000, resampler=context.resampler)
        is_speech, vad_prob = await asyncio.to_thread(self.vad.is_speech, chunk_16k)
        chunk.set_feature("analysis_view_16k", chunk_16k)
        chunk.set_feature("is_speech", is_speech)
        chunk.set_feature("vad_prob", vad_prob)


class PassthroughPolicy(FramePolicy):
    def decide(self, chunk: InputChunk, context: ExecutionContext) -> GateDecision:
        block_16k = chunk.get_feature("analysis_view_16k")
        return GateDecision(
            action="run_rvc",
            engine_input=chunk.master_samples,
            block_16k=block_16k,
        )


class SpeechGatePolicy(FramePolicy):
    def decide(self, chunk: InputChunk, context: ExecutionContext) -> GateDecision:
        block_16k = chunk.get_feature("analysis_view_16k")
        if chunk.get_feature("is_speech") is False:
            silence_input = np.zeros(chunk.master_samples.shape[0], dtype=np.float32)
            silence_output = np.zeros(chunk.master_samples.shape[0], dtype=np.float32)
            return GateDecision(
                action="push_silence",
                engine_input=silence_input,
                block_16k=block_16k,
                output_override=silence_output,
            )
        return GateDecision(
            action="run_rvc",
            engine_input=chunk.master_samples,
            block_16k=block_16k,
        )


class RealtimeSession:
    def __init__(self):
        self.status = SessionStatus.IDLE
        self.transport_status = TransportStatus.DETACHED
        self.engine: Optional[RealtimeRVCEngine] = None
        self.adapter: Optional[RVCRealtimeAdapter] = None
        self.input_queue: asyncio.Queue[InputChunk] | None = None
        self.output_slot = OutputSlot()
        self.output_event = asyncio.Event()
        self.slot_lock = asyncio.Lock()
        self.session_lock = asyncio.Lock()
        self.infer_task: asyncio.Task | None = None
        self.active_websocket: WebSocket | None = None
        self.stats = SessionStats()
        self.created_at = 0.0
        self.last_attach_at = 0.0
        self.last_detach_at = 0.0
        self.reconnect_deadline = 0.0
        self.input_sequence = 0
        self.output_version = 0
        self.stop_requested = False
        self.flush_requested = False
        self.input_ttl_ms = 500.0
        self.output_ttl_ms = 500.0
        self.reconnect_ttl_ms = 30000.0
        self.vad: Optional[SileroVAD] = None
        self.analyzers: list[FrameAnalyzer] = []
        self.policy: FramePolicy = PassthroughPolicy()
        self.executor = FrameExecutor()

    @staticmethod
    def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return samples.astype(np.float32, copy=False)
        return librosa.resample(
            samples.astype(np.float32, copy=False),
            orig_sr=orig_sr,
            target_sr=target_sr,
        ).astype(np.float32, copy=False)

    async def start(self, req: StartRequest):
        async with self.session_lock:
            await self._stop_locked()

            original_argv = sys.argv[:]
            try:
                sys.argv = [sys.argv[0]]
                config = Config()
            finally:
                sys.argv = original_argv
            config.use_jit = False

            model_config = RVCModelConfig(
                pth_path=req.pth_path,
                index_path=req.index_path,
                index_rate=req.index_rate,
                pitch_shift=req.pitch,
                formant_shift=req.formant,
            )
            self.adapter = RVCRealtimeAdapter(model_config, config)
            input_sr = req.input_sr or self.adapter.tgt_sr
            self.engine = RealtimeRVCEngine(
                self.adapter,
                RealtimeRVCConfig(
                    block_time=req.block_time,
                    crossfade_time=req.crossfade_time,
                    extra_time=req.extra_time,
                    use_phase_vocoder=req.use_phase_vocoder,
                ),
                input_sr=input_sr,
            )
            self.engine.initialize_stream()
            if req.enable_vad:
                device_str = (
                    str(self.adapter.device) if self.adapter is not None else "cpu"
                )
                self.vad = SileroVAD(threshold=req.vad_threshold, device=device_str)
                await asyncio.to_thread(self.vad.warmup)
                self.analyzers = [SileroVADAnalyzer(self.vad)]
                self.policy = SpeechGatePolicy()
            else:
                self.vad = None
                self.analyzers = []
                self.policy = PassthroughPolicy()
            self.input_queue = asyncio.Queue(maxsize=max(1, req.input_queue_size))
            self.output_slot = OutputSlot()
            self.output_event = asyncio.Event()
            self.stats = SessionStats()
            self.status = SessionStatus.RUNNING
            self.transport_status = TransportStatus.DETACHED
            self.created_at = time.monotonic()
            self.last_detach_at = self.created_at
            self.last_attach_at = 0.0
            self.reconnect_ttl_ms = req.reconnect_ttl_ms
            self.reconnect_deadline = self.created_at + req.reconnect_ttl_ms / 1000.0
            self.input_ttl_ms = req.input_ttl_ms
            self.output_ttl_ms = req.output_ttl_ms
            self.input_sequence = 0
            self.output_version = 0
            self.stop_requested = False
            self.flush_requested = False
            self.infer_task = asyncio.create_task(self._infer_loop(), name="rvc-infer")

    async def stop(self):
        async with self.session_lock:
            await self._stop_locked()

    async def _stop_locked(self):
        self.stop_requested = True
        self.flush_requested = False
        if self.infer_task is not None:
            self.infer_task.cancel()
            try:
                await self.infer_task
            except asyncio.CancelledError:
                pass
            self.infer_task = None
        self.status = (
            SessionStatus.STOPPED
            if self.engine is not None or self.adapter is not None
            else SessionStatus.IDLE
        )
        self.transport_status = TransportStatus.DETACHED
        self.active_websocket = None
        self.input_queue = None
        self.engine = None
        self.adapter = None
        if self.vad is not None:
            self.vad.reset()
            self.vad = None
        self.analyzers = []
        self.policy = PassthroughPolicy()
        self.output_event = asyncio.Event()
        self.output_slot = OutputSlot()

    async def attach(self, websocket: WebSocket):
        async with self.session_lock:
            if self.status not in (SessionStatus.RUNNING, SessionStatus.FLUSHING):
                await websocket.close(code=1013, reason="session not running")
                return False
            if self.active_websocket is not None:
                await self.active_websocket.close(
                    code=1012, reason="replaced by new connection"
                )
            self.active_websocket = websocket
            self.transport_status = TransportStatus.ATTACHED
            self.last_attach_at = time.monotonic()
            return True

    async def detach(self, websocket: WebSocket | None = None):
        async with self.session_lock:
            if websocket is not None and self.active_websocket is not websocket:
                return
            self.active_websocket = None
            self.transport_status = TransportStatus.DETACHED
            self.last_detach_at = time.monotonic()
            self.reconnect_deadline = (
                self.last_detach_at + self.reconnect_ttl_ms / 1000.0
            )

    async def submit_input(self, samples: np.ndarray):
        if self.status not in (SessionStatus.RUNNING, SessionStatus.FLUSHING):
            return False
        if self.input_queue is None:
            return False
        now = time.monotonic()
        chunk = InputChunk(
            sequence_id=self.input_sequence,
            input_sr=self.engine.input_sr if self.engine is not None else 0,
            master_samples=samples.astype(np.float32, copy=False),
            sample_views={},
            features={},
            created_at=now,
            expire_at=now + self.input_ttl_ms / 1000.0,
        )
        self.input_sequence += 1

        while self.input_queue.full():
            try:
                self.input_queue.get_nowait()
                self.input_queue.task_done()
                self.stats.dropped_input_chunks += 1
            except asyncio.QueueEmpty:
                break
        try:
            self.input_queue.put_nowait(chunk)
            self.stats.accepted_input_chunks += 1
            return True
        except asyncio.QueueFull:
            self.stats.dropped_input_chunks += 1
            return False

    async def request_flush(self):
        if self.status == SessionStatus.RUNNING:
            self.flush_requested = True
            self.status = SessionStatus.FLUSHING

    async def _publish_output(
        self,
        sequence_id: int,
        wave: np.ndarray,
        infer_time_ms: float,
        vad_prob: float | None = None,
    ):
        now = time.monotonic()
        expire_at = now + self.output_ttl_ms / 1000.0
        payload = wave.astype(np.float32, copy=False).tobytes()
        async with self.slot_lock:
            old = self.output_slot
            if old.version > 0 and old.expire_at > now:
                self.stats.dropped_output_chunks += 1
            self.output_version += 1
            self.output_slot = OutputSlot(
                version=self.output_version,
                sequence_id=sequence_id,
                payload=payload,
                sample_count=wave.shape[0],
                created_at=now,
                expire_at=expire_at,
                infer_time_ms=infer_time_ms,
                vad_prob=vad_prob,
            )
            self.stats.produced_output_chunks += 1
            self.stats.last_output_sequence_id = sequence_id
            self.stats.last_infer_time_ms = infer_time_ms
        self.output_event.set()

    async def _infer_loop(self):
        try:
            while not self.stop_requested:
                now = time.monotonic()
                if (
                    self.transport_status == TransportStatus.DETACHED
                    and now > self.reconnect_deadline
                ):
                    self.status = SessionStatus.STOPPED
                    break

                chunk: InputChunk | None = None
                if self.input_queue is not None:
                    try:
                        chunk = await asyncio.wait_for(
                            self.input_queue.get(), timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        chunk = None

                if chunk is None:
                    if self.flush_requested and self.engine is not None:
                        remaining = self.engine.get_algorithmic_delay_samples()
                        self.flush_requested = False
                        while remaining > 0 and not self.stop_requested:
                            current = min(self.engine.block_frame, remaining)
                            silence = np.zeros(current, dtype=np.float32)
                            result = await asyncio.to_thread(
                                self.engine.process_block,
                                silence,
                                silence_16k=self._resample(silence, self.engine.input_sr, 16000),
                            )
                            await self._publish_output(
                                result.block_index,
                                result.output_wave[:current],
                                result.infer_time_ms,
                            )
                            remaining -= current
                        self.status = SessionStatus.RUNNING
                    continue

                try:
                    if self.input_queue is not None:
                        self.input_queue.task_done()
                except ValueError:
                    pass

                if chunk.expire_at < time.monotonic():
                    self.stats.expired_input_chunks += 1
                    continue

                if self.engine is None:
                    continue

                analysis_context = AnalysisContext(resampler=self._resample)
                for analyzer in self.analyzers:
                    await analyzer.analyze(chunk, analysis_context)

                execution_context = ExecutionContext(
                    engine=self.engine,
                    resampler=self._resample,
                )
                decision = self.policy.decide(chunk, execution_context)
                output_wave, infer_time_ms = await self.executor.execute(
                    chunk,
                    decision,
                    execution_context,
                )
                await self._publish_output(
                    chunk.sequence_id,
                    output_wave,
                    infer_time_ms,
                    vad_prob=chunk.get_feature("vad_prob"),
                )
        except asyncio.CancelledError:
            raise
        finally:
            if self.status != SessionStatus.STOPPED:
                self.status = SessionStatus.IDLE

    async def get_state(self):
        now = time.monotonic()
        async with self.slot_lock:
            slot = self.output_slot
            if slot.version > 0 and slot.expire_at < now:
                self.stats.expired_output_chunks += 1
                self.output_slot = OutputSlot()
                slot = self.output_slot
        queue_size = self.input_queue.qsize() if self.input_queue is not None else 0
        congested = queue_size > 0 and self.stats.dropped_input_chunks > 0
        self.stats.congested = congested
        return {
            "status": self.status.value,
            "transport_status": self.transport_status.value,
            "input_queue_depth": queue_size,
            "has_fresh_output": slot.version > 0 and slot.expire_at >= now,
            "stats": asdict(self.stats),
            "reconnect_deadline": self.reconnect_deadline,
        }


session = RealtimeSession()
app = FastAPI(title="realtime_v2 RVC server")


@app.post("/realtime/start")
async def start_realtime(req: StartRequest):
    await session.start(req)
    return {"ok": True}


@app.post("/realtime/stop")
async def stop_realtime():
    await session.stop()
    return {"ok": True}


@app.get("/realtime/state")
async def get_realtime_state():
    return await session.get_state()


@app.get("/realtime/rtt")
async def get_rtt_probe(ts: float, seq: int = 0):
    return {
        "type": "rtt_probe_response",
        "seq": seq,
        "client_ts": ts,
        "server_ts": time.perf_counter(),
        "server_wall_time": time.time(),
    }


@app.post("/realtime/rtt")
async def post_rtt_probe(req: RttProbeRequest):
    return {
        "type": "rtt_probe_response",
        "seq": req.seq,
        "client_ts": req.client_ts,
        "server_ts": time.perf_counter(),
        "server_wall_time": time.time(),
    }


async def _send_loop(websocket: WebSocket):
    last_sent_version = 0
    while True:
        await session.output_event.wait()
        session.output_event.clear()
        now = time.monotonic()
        async with session.slot_lock:
            slot = session.output_slot
            if slot.version <= last_sent_version:
                continue
            if slot.expire_at < now:
                session.stats.expired_output_chunks += 1
                session.output_slot = OutputSlot()
                continue
            version = slot.version
            payload = slot.payload
            meta = {
                "type": "audio_chunk",
                "sequence_id": slot.sequence_id,
                "sample_count": slot.sample_count,
                "infer_time_ms": slot.infer_time_ms,
                "version": slot.version,
            }
        if slot.vad_prob is not None:
            meta["vad_prob"] = slot.vad_prob
        await websocket.send_text(json.dumps(meta))
        await websocket.send_bytes(payload)
        last_sent_version = version


async def _recv_loop(websocket: WebSocket):
    while True:
        message = await websocket.receive()
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            raise WebSocketDisconnect()
        if message_type == "websocket.receive":
            text = message.get("text")
            if text is not None:
                data = json.loads(text)
                msg_type = data.get("type")
                if msg_type == "flush":
                    await session.request_flush()
                elif msg_type == "stop":
                    await session.stop()
                    return
                elif msg_type == "ping":
                    await websocket.send_text(
                        json.dumps({"type": "pong", "ts": time.time()})
                    )
                elif msg_type == "rtt_probe":
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "rtt_probe_response",
                                "seq": int(data.get("seq", 0)),
                                "client_ts": float(data.get("client_ts", 0.0)),
                                "server_ts": time.perf_counter(),
                                "server_wall_time": time.time(),
                            }
                        )
                    )
                continue
            body = message.get("bytes")
            if body is not None:
                samples = np.frombuffer(body, dtype=np.float32).copy()
                await session.submit_input(samples)


@app.websocket("/realtime/ws")
async def realtime_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        attached = await session.attach(websocket)
        if not attached:
            return
        await websocket.send_text(json.dumps({"type": "attached"}))
        recv_task = asyncio.create_task(_recv_loop(websocket), name="recv-realtime")
        send_task = asyncio.create_task(_send_loop(websocket), name="send-realtime")
        done, pending = await asyncio.wait(
            {recv_task, send_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for task in done:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                exc = None
            if exc is not None:
                raise exc
    except WebSocketDisconnect:
        pass
    finally:
        await session.detach(websocket)


def main():
    parser = argparse.ArgumentParser(
        description="FastAPI realtime server for realtime_v2 RVC"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6243)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
