from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import numpy as np
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from realtime_v2.engine import RealtimeRVCConfig, RealtimeRVCEngine
from realtime_v2.model_adapter import RVCModelConfig, RVCRealtimeAdapter
from realtime_v2.runtime import AnalysisContext, ExecutionContext, InputChunk, RuntimePipeline, build_runtime


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
    enable_dfn: bool = False
    dfn_backend: str = "native"
    dfn_atten_lim: float | None = None
    dfn_post_filter_beta: float = 0.0
    dfn_compensate_delay: bool = False


class RttProbeRequest(BaseModel):
    client_ts: float
    seq: int = 0


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
    dfn_metrics: dict | None = None
    timings: dict | None = None


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
    last_input_queue_wait_ms: float = 0.0
    last_analysis_time_ms: float = 0.0
    last_policy_time_ms: float = 0.0
    last_execute_time_ms: float = 0.0
    last_publish_to_send_wait_ms: float = 0.0
    last_send_time_ms: float = 0.0
    last_end_to_end_server_ms: float = 0.0
    max_input_queue_wait_ms: float = 0.0
    max_analysis_time_ms: float = 0.0
    max_policy_time_ms: float = 0.0
    max_execute_time_ms: float = 0.0
    max_publish_to_send_wait_ms: float = 0.0
    max_send_time_ms: float = 0.0
    max_end_to_end_server_ms: float = 0.0


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
        self.runtime: RuntimePipeline | None = None
        self.dfn_model_path: str | None = None

    @staticmethod
    def _update_timing_stat(stats: SessionStats, name: str, value: float):
        setattr(stats, f"last_{name}", value)
        max_name = f"max_{name}"
        current_max = getattr(stats, max_name)
        if value > current_max:
            setattr(stats, max_name, value)

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
            self.runtime = await build_runtime(
                req,
                self.adapter,
                dfn_model_path=self.dfn_model_path,
            )
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
        if self.runtime is not None and self.runtime.vad is not None:
            self.runtime.vad.reset()
        if self.runtime is not None and self.runtime.dfn is not None:
            self.runtime.dfn.close()
        self.runtime = None
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
            streams={},
            features={},
            created_at=now,
            expire_at=now + self.input_ttl_ms / 1000.0,
        )
        chunk.set_feature("submitted_at", now)
        chunk.set_stream(
            "input",
            samples.astype(np.float32, copy=False),
            self.engine.input_sr if self.engine is not None else 0,
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
        dfn_metrics: dict | None = None,
        timings: dict | None = None,
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
                dfn_metrics=dfn_metrics,
                timings=timings,
            )
            self.stats.produced_output_chunks += 1
            self.stats.last_output_sequence_id = sequence_id
            self.stats.last_infer_time_ms = infer_time_ms
            if timings is not None:
                for name in (
                    "input_queue_wait_ms",
                    "analysis_time_ms",
                    "policy_time_ms",
                    "execute_time_ms",
                    "server_pipeline_ms",
                ):
                    value = float(timings.get(name, 0.0))
                    self._update_timing_stat(self.stats, name, value)
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

                if self.runtime is None:
                    continue

                dequeued_at = time.monotonic()
                submitted_at = float(chunk.get_feature("submitted_at", chunk.created_at))
                input_queue_wait_ms = max(0.0, (dequeued_at - submitted_at) * 1000.0)

                analysis_context = AnalysisContext(resampler=self._resample)
                analyzer_timings: dict[str, float] = {}
                analysis_start = time.monotonic()
                for analyzer in self.runtime.analyzers:
                    analyzer_name = analyzer.__class__.__name__
                    analyzer_start = time.monotonic()
                    await analyzer.analyze(chunk, analysis_context)
                    analyzer_timings[analyzer_name] = (
                        time.monotonic() - analyzer_start
                    ) * 1000.0
                analysis_time_ms = (time.monotonic() - analysis_start) * 1000.0

                execution_context = ExecutionContext(
                    engine=self.engine,
                    resampler=self._resample,
                )
                policy_start = time.monotonic()
                decision = self.runtime.policy.decide(chunk, execution_context)
                policy_time_ms = (time.monotonic() - policy_start) * 1000.0
                execute_start = time.monotonic()
                output_wave, infer_time_ms = await self.runtime.executor.execute(
                    chunk,
                    decision,
                    execution_context,
                )
                execute_time_ms = (time.monotonic() - execute_start) * 1000.0
                publish_requested_at = time.monotonic()
                timings = {
                    "submitted_at": submitted_at,
                    "dequeued_at": dequeued_at,
                    "input_queue_wait_ms": input_queue_wait_ms,
                    "analysis_time_ms": analysis_time_ms,
                    "policy_time_ms": policy_time_ms,
                    "execute_time_ms": execute_time_ms,
                    "server_pipeline_ms": (publish_requested_at - submitted_at) * 1000.0,
                    "analyzers": analyzer_timings,
                }
                await self._publish_output(
                    chunk.sequence_id,
                    output_wave,
                    infer_time_ms,
                    vad_prob=chunk.get_feature("vad_prob"),
                    dfn_metrics={
                        "enabled": chunk.get_feature("dfn_enabled", False),
                        "input_rms": chunk.get_feature("dfn_input_rms"),
                        "output_rms": chunk.get_feature("dfn_output_rms"),
                        "residual_rms": chunk.get_feature("dfn_residual_rms"),
                        "rms_ratio": chunk.get_feature("dfn_rms_ratio"),
                    },
                    timings=timings,
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
            timings = dict(slot.timings or {})
        if slot.vad_prob is not None:
            meta["vad_prob"] = slot.vad_prob
        if slot.dfn_metrics is not None:
            meta["dfn"] = slot.dfn_metrics
        send_started_at = time.monotonic()
        if timings:
            publish_created_at = float(timings.get("published_at", slot.created_at))
            timings["publish_to_send_wait_ms"] = max(
                0.0, (send_started_at - publish_created_at) * 1000.0
            )
        await websocket.send_text(json.dumps(meta))
        await websocket.send_bytes(payload)
        send_finished_at = time.monotonic()
        if timings:
            timings["send_time_ms"] = (send_finished_at - send_started_at) * 1000.0
            submitted_at = float(timings.get("submitted_at", send_started_at))
            timings["end_to_end_server_ms"] = (
                send_finished_at - submitted_at
            ) * 1000.0
            meta["timings"] = timings
            session._update_timing_stat(
                session.stats,
                "publish_to_send_wait_ms",
                float(timings["publish_to_send_wait_ms"]),
            )
            session._update_timing_stat(
                session.stats,
                "send_time_ms",
                float(timings["send_time_ms"]),
            )
            session._update_timing_stat(
                session.stats,
                "end_to_end_server_ms",
                float(timings["end_to_end_server_ms"]),
            )
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
    parser.add_argument("--dfn-model-path", type=str, default=None)
    args = parser.parse_args()
    session.dfn_model_path = args.dfn_model_path
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
