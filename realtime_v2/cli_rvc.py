from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf
from websockets.exceptions import ConnectionClosed

try:
    from sounddevice import WasapiSettings
except ImportError:
    WasapiSettings = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _format_latency_ms(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) > 0:
        return float(value[-1]) * 1000.0
    return float(value) * 1000.0


def parse_device_argument(value: str | None):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def create_sounddevice_extra_settings(low_latency_wasapi: bool, exclusive_wasapi: bool):
    if not low_latency_wasapi or WasapiSettings is None:
        return None
    return (
        WasapiSettings(exclusive=bool(exclusive_wasapi)),
        WasapiSettings(exclusive=bool(exclusive_wasapi)),
    )


def list_hostapis():
    sd._terminate()
    sd._initialize()
    hostapis = sd.query_hostapis()
    for index, hostapi in enumerate(hostapis):
        default_in = hostapi.get("default_input_device", -1)
        default_out = hostapi.get("default_output_device", -1)
        device_count = len(hostapi.get("devices", []))
        print(
            f"[{index}] {hostapi['name']} | devices={device_count} "
            f"default_input={default_in} default_output={default_out}"
        )


def list_audio_devices(
    hostapi_filter: str | None = None,
    direction: str | None = None,
):
    """列出音频设备。

    Args:
        hostapi_filter: 按 hostapi 名称模糊过滤（大小写不敏感）。
        direction: 'input'  -> 仅显示有输入声道的设备（可用作麦克风/采集）；
                   'output' -> 仅显示有输出声道的设备（可用作扬声器/播放）；
                   None     -> 显示全部。
    """
    sd._terminate()
    sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    hostapi_names = {idx: hostapi["name"] for idx, hostapi in enumerate(hostapis)}

    filter_lower = hostapi_filter.lower().strip() if hostapi_filter else None

    for index, device in enumerate(devices):
        max_in = device.get("max_input_channels", 0)
        max_out = device.get("max_output_channels", 0)
        if direction == "input" and max_in <= 0:
            continue
        if direction == "output" and max_out <= 0:
            continue
        hostapi_idx = device.get("hostapi", -1)
        hostapi_name = hostapi_names.get(hostapi_idx, "unknown")
        if filter_lower is not None and filter_lower not in hostapi_name.lower():
            continue
        print(
            f"[{index}] {device['name']} | hostapi={hostapi_name} | "
            f"max_input_ch={max_in} max_output_ch={max_out} "
            f"default_sr={device['default_samplerate']}"
        )


@dataclass
class RuntimeStats:
    sent_chunks: int = 0
    sent_samples: int = 0
    dropped_input_chunks: int = 0
    received_chunks: int = 0
    received_samples: int = 0
    dropped_output_chunks: int = 0
    received_events: int = 0
    reconnect_count: int = 0
    input_overflow_count: int = 0
    input_underflow_count: int = 0
    output_underflow_count: int = 0
    output_overflow_count: int = 0
    dropped_input_blocks_for_latency: int = 0
    dropped_output_blocks_for_latency: int = 0


@dataclass
class RVCChunkTrace:
    sequence_id: int
    sent_at: float
    sample_count: int


@dataclass
class RVCPerformanceStats:
    infer_count: int = 0
    total_infer_ms: float = 0.0
    max_infer_ms: float = 0.0
    last_infer_ms: float = 0.0
    e2e_count: int = 0
    total_e2e_ms: float = 0.0
    max_e2e_ms: float = 0.0
    last_e2e_ms: float = 0.0
    # VAD 
    vad_chunks_total: int = 0
    vad_chunks_speech: int = 0
    vad_chunks_silence: int = 0
    vad_prob_sum: float = 0.0
    vad_prob_last: float | None = None
    vad_prob_min: float = 1.0
    vad_prob_max: float = 0.0

    def record(self, infer_time_ms: float | None, e2e_ms: float | None):
        if infer_time_ms is not None:
            infer_value = float(infer_time_ms)
            self.infer_count += 1
            self.total_infer_ms += infer_value
            self.last_infer_ms = infer_value
            self.max_infer_ms = max(self.max_infer_ms, infer_value)
        if e2e_ms is not None:
            e2e_value = float(e2e_ms)
            self.e2e_count += 1
            self.total_e2e_ms += e2e_value
            self.last_e2e_ms = e2e_value
            self.max_e2e_ms = max(self.max_e2e_ms, e2e_value)

    def record_vad(self, vad_prob: float, is_silence: bool):
        """记录一个 chunk 的 VAD 指标（infer_time_ms=0 时为 silence chunk）。"""
        self.vad_chunks_total += 1
        self.vad_prob_sum += vad_prob
        self.vad_prob_last = vad_prob
        self.vad_prob_min = min(self.vad_prob_min, vad_prob)
        self.vad_prob_max = max(self.vad_prob_max, vad_prob)
        if is_silence:
            self.vad_chunks_silence += 1
        else:
            self.vad_chunks_speech += 1

    def to_summary(self):
        summary = {
            "infer_count": self.infer_count,
            "infer_avg_ms": (
                round(self.total_infer_ms / self.infer_count, 3)
                if self.infer_count
                else None
            ),
            "infer_max_ms": round(self.max_infer_ms, 3) if self.infer_count else None,
            "infer_last_ms": round(self.last_infer_ms, 3) if self.infer_count else None,
            "e2e_count": self.e2e_count,
            "e2e_avg_ms": (
                round(self.total_e2e_ms / self.e2e_count, 3) if self.e2e_count else None
            ),
            "e2e_max_ms": round(self.max_e2e_ms, 3) if self.e2e_count else None,
            "e2e_last_ms": round(self.last_e2e_ms, 3) if self.e2e_count else None,
        }
        if self.vad_chunks_total > 0:
            summary["vad"] = {
                "total_chunks": self.vad_chunks_total,
                "speech_chunks": self.vad_chunks_speech,
                "silence_chunks": self.vad_chunks_silence,
                "silence_ratio": round(self.vad_chunks_silence / self.vad_chunks_total, 4),
                "prob_avg": round(self.vad_prob_sum / self.vad_chunks_total, 4),
                "prob_min": round(self.vad_prob_min, 4),
                "prob_max": round(self.vad_prob_max, 4),
                "prob_last": round(self.vad_prob_last, 4) if self.vad_prob_last is not None else None,
            }
        return summary


class RVCPerformanceLogger:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.pending_chunks: collections.OrderedDict[int, RVCChunkTrace] = (
            collections.OrderedDict()
        )
        self.stats = RVCPerformanceStats()

    def track_sent_chunk(
        self, sequence_id: int, sample_count: int, sent_at: float | None = None
    ):
        if not self.enabled:
            return
        self.pending_chunks[int(sequence_id)] = RVCChunkTrace(
            sequence_id=int(sequence_id),
            sent_at=time.perf_counter() if sent_at is None else float(sent_at),
            sample_count=int(sample_count),
        )

    def handle_audio_meta(self, payload: dict):
        if not self.enabled:
            return None
        sequence_id = payload.get("sequence_id")
        if sequence_id is None:
            return None
        try:
            sequence_id = int(sequence_id)
        except (TypeError, ValueError):
            return None

        infer_raw = payload.get("infer_time_ms")
        infer_time_ms = None
        try:
            if infer_raw is not None:
                infer_time_ms = float(infer_raw)
        except (TypeError, ValueError):
            infer_time_ms = None

        trace = self.pending_chunks.pop(sequence_id, None)
        e2e_ms = None
        sent_sample_count = None
        if trace is not None:
            e2e_ms = (time.perf_counter() - trace.sent_at) * 1000.0
            sent_sample_count = trace.sample_count
            stale_keys = [
                key for key in self.pending_chunks.keys() if key < sequence_id
            ]
            for key in stale_keys:
                self.pending_chunks.pop(key, None)

        self.stats.record(infer_time_ms, e2e_ms)

        vad_prob_raw = payload.get("vad_prob")
        vad_prob: float | None = None
        if vad_prob_raw is not None:
            try:
                vad_prob = float(vad_prob_raw)
            except (TypeError, ValueError):
                vad_prob = None
        if vad_prob is not None:
            is_silence = (infer_time_ms is not None and infer_time_ms == 0.0)
            self.stats.record_vad(vad_prob, is_silence)

        sample_count = payload.get("sample_count", "?")
        infer_display = f"{infer_time_ms:.2f}" if infer_time_ms is not None else "?"
        e2e_display = f"{e2e_ms:.2f}" if e2e_ms is not None else "?"
        sent_display = sent_sample_count if sent_sample_count is not None else "?"
        vad_display = f" vad_prob={vad_prob:.3f}" if vad_prob is not None else ""
        print(
            f"[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}] rvc-meta seq={sequence_id} infer_ms={infer_display} e2e_ms={e2e_display} sample_count={sample_count} sent_samples={sent_display}{vad_display}",
            flush=True,
        )
        return {
            "sequence_id": sequence_id,
            "infer_time_ms": infer_time_ms,
            "e2e_ms": e2e_ms,
            "vad_prob": vad_prob,
        }

    def get_summary(self):
        if not self.enabled:
            return None
        summary = self.stats.to_summary()
        summary["pending_traces"] = len(self.pending_chunks)
        return summary


def recommend_runtime_parameters(
    rtt_ms: float | None, block_time: float, reconnect_delay_ms: float
):
    if rtt_ms is None:
        return {
            "recommended_input_ttl_ms": 500.0,
            "recommended_output_ttl_ms": 500.0,
            "recommended_reconnect_delay_ms": reconnect_delay_ms,
            "recommended_input_queue_size": 8,
            "recommended_client_output_queue_blocks": 8,
            "note": "未完成 RTT 测试，使用保守默认值。",
        }

    block_ms = block_time * 1000.0
    jitter_budget_ms = max(40.0, rtt_ms * 0.5)
    input_ttl_ms = max(2.0 * block_ms, rtt_ms + 2.0 * jitter_budget_ms)
    output_ttl_ms = max(block_ms, 0.75 * rtt_ms + jitter_budget_ms)
    reconnect_delay_ms = max(250.0, min(reconnect_delay_ms, max(500.0, rtt_ms * 1.5)))
    input_queue_size = max(4, int(np.ceil(input_ttl_ms / block_ms)))
    output_queue_blocks = max(3, int(np.ceil(output_ttl_ms / block_ms)))
    return {
        "recommended_input_ttl_ms": round(float(input_ttl_ms), 2),
        "recommended_output_ttl_ms": round(float(output_ttl_ms), 2),
        "recommended_reconnect_delay_ms": round(float(reconnect_delay_ms), 2),
        "recommended_input_queue_size": int(input_queue_size),
        "recommended_client_output_queue_blocks": int(output_queue_blocks),
        "note": "建议值按 RTT 和 block_time 估算，链路抖动更大时应进一步调高 TTL。",
    }


async def measure_http_rtt(base_url: str, count: int, timeout: float):
    if count <= 0:
        return None
    samples = []
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        for seq in range(count):
            client_ts = time.perf_counter()
            resp = await client.get(
                "/realtime/rtt", params={"ts": client_ts, "seq": seq}
            )
            resp.raise_for_status()
            _ = resp.json()
            now = time.perf_counter()
            samples.append((now - client_ts) * 1000.0)
    return {
        "count": len(samples),
        "min_ms": round(float(min(samples)), 3),
        "avg_ms": round(float(sum(samples) / len(samples)), 3),
        "max_ms": round(float(max(samples)), 3),
    }


async def measure_ws_rtt(ws, count: int, interval: float):
    if count <= 0:
        return None
    samples = []
    for seq in range(count):
        client_ts = time.perf_counter()
        await ws.send(
            json.dumps({"type": "rtt_probe", "seq": seq, "client_ts": client_ts})
        )
        while True:
            message = await ws.recv()
            if not isinstance(message, str):
                continue
            payload = json.loads(message)
            if (
                payload.get("type") != "rtt_probe_response"
                or int(payload.get("seq", -1)) != seq
            ):
                continue
            now = time.perf_counter()
            samples.append((now - client_ts) * 1000.0)
            break
        if interval > 0:
            await asyncio.sleep(interval)
    return {
        "count": len(samples),
        "min_ms": round(float(min(samples)), 3),
        "avg_ms": round(float(sum(samples) / len(samples)), 3),
        "max_ms": round(float(max(samples)), 3),
    }


def load_mono_audio_with_sr(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def build_start_payload(args):
    return {
        "pth_path": args.pth,
        "index_path": args.index,
        "index_rate": args.index_rate,
        "pitch": args.pitch,
        "formant": args.formant,
        "block_time": args.block_time,
        "crossfade_time": args.crossfade_time,
        "extra_time": args.extra_time,
        "input_sr": args.input_sr,
        "input_queue_size": args.input_queue_size,
        "input_ttl_ms": args.input_ttl_ms,
        "output_ttl_ms": args.output_ttl_ms,
        "reconnect_ttl_ms": args.reconnect_ttl_ms,
        "use_phase_vocoder": args.use_phase_vocoder,
        "enable_vad": args.silero_vad
    }


def _seconds_to_aligned_samples(seconds: float, sample_rate: int) -> int:
    zc = sample_rate // 100
    return int(np.round(seconds * sample_rate / zc)) * zc


def get_rvc_theoretical_latency_summary(
    input_sr: int, block_time: float, crossfade_time: float, extra_time: float
):
    if input_sr <= 0:
        raise ValueError("input_sr must be positive")

    zc = input_sr // 100
    block_frame = _seconds_to_aligned_samples(block_time, input_sr)
    crossfade_frame = _seconds_to_aligned_samples(crossfade_time, input_sr)
    sola_search_frame = zc
    sola_buffer_frame = min(crossfade_frame, 4 * zc)
    extra_frame = _seconds_to_aligned_samples(extra_time, input_sr)

    algorithmic_delay_samples = crossfade_frame + sola_search_frame
    algorithmic_delay_ms = algorithmic_delay_samples * 1000.0 / input_sr
    input_window_samples = (
        extra_frame + crossfade_frame + sola_search_frame + block_frame
    )
    input_window_ms = input_window_samples * 1000.0 / input_sr
    block_ms = block_frame * 1000.0 / input_sr
    capture_worst_ms = block_ms
    capture_avg_ms = block_ms * 0.5

    return {
        "input_sr": int(input_sr),
        "block_frame": int(block_frame),
        "block_ms": round(block_ms, 3),
        "crossfade_frame": int(crossfade_frame),
        "crossfade_ms": round(crossfade_frame * 1000.0 / input_sr, 3),
        "sola_search_frame": int(sola_search_frame),
        "sola_search_ms": round(sola_search_frame * 1000.0 / input_sr, 3),
        "sola_buffer_frame": int(sola_buffer_frame),
        "sola_buffer_ms": round(sola_buffer_frame * 1000.0 / input_sr, 3),
        "extra_frame": int(extra_frame),
        "extra_ms": round(extra_frame * 1000.0 / input_sr, 3),
        "algorithmic_delay_samples": int(algorithmic_delay_samples),
        "algorithmic_delay_ms": round(algorithmic_delay_ms, 3),
        "input_window_samples": int(input_window_samples),
        "input_window_ms": round(input_window_ms, 3),
        "capture_avg_ms": round(capture_avg_ms, 3),
        "capture_worst_ms": round(capture_worst_ms, 3),
        "note": "algorithmic_delay_ms 仅包含 RVC 引擎内 crossfade + SOLA search 的理论算法延迟；主观 offset 还会叠加采集成块、推理、网络、播放缓冲，以及 extra_time 带来的长上下文响应惯性。",
    }


def print_rvc_theoretical_latency(args, input_sr: int | None = None):
    sr = int(input_sr if input_sr is not None else args.input_sr)
    if sr <= 0:
        return None
    summary = get_rvc_theoretical_latency_summary(
        sr, args.block_time, args.crossfade_time, args.extra_time
    )
    print("RVC theoretical latency:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


async def run_rtt_test_mode(args):
    http_stats = await measure_http_rtt(
        args.base_url, args.http_rtt_count, args.http_timeout
    )
    if http_stats is not None:
        print("HTTP RTT:")
        print(json.dumps(http_stats, ensure_ascii=False, indent=2))

    if args.ws_rtt_count <= 0:
        recommended = recommend_runtime_parameters(
            None, args.block_time, args.reconnect_delay_ms
        )
        print("Recommended runtime parameters:")
        print(json.dumps(recommended, ensure_ascii=False, indent=2))
        return

    import websockets

    ws_url = (
        args.base_url.replace("http://", "ws://").replace("https://", "wss://")
        + "/realtime/ws"
    )
    try:
        if args.auto_start_for_rtt:
            start_payload = build_start_payload(args)
            async with httpx.AsyncClient(
                base_url=args.base_url, timeout=max(30.0, args.http_timeout)
            ) as client:
                resp = await client.post("/realtime/start", json=start_payload)
                resp.raise_for_status()
            print_rvc_theoretical_latency(args)
        async with websockets.connect(ws_url, max_size=None) as ws:
            ws_stats = await measure_ws_rtt(ws, args.ws_rtt_count, args.ws_rtt_interval)
    except Exception:
        ws_stats = None
    finally:
        if args.auto_start_for_rtt:
            async with httpx.AsyncClient(
                base_url=args.base_url, timeout=max(30.0, args.http_timeout)
            ) as client:
                try:
                    await client.post("/realtime/stop")
                except Exception:
                    pass

    if ws_stats is not None:
        print("WebSocket RTT:")
        print(json.dumps(ws_stats, ensure_ascii=False, indent=2))
        recommended = recommend_runtime_parameters(
            ws_stats["avg_ms"], args.block_time, args.reconnect_delay_ms
        )
    else:
        print("WebSocket RTT: unavailable")
        recommended = recommend_runtime_parameters(
            None, args.block_time, args.reconnect_delay_ms
        )
    print("Recommended runtime parameters:")
    print(json.dumps(recommended, ensure_ascii=False, indent=2))


async def run_offline_mode(args):
    if not args.source:
        raise ValueError("offline 模式需要 --source")
    if not args.output:
        raise ValueError("offline 模式需要 --output")

    args.output = os.path.abspath(args.output)
    source_wave, source_sr = load_mono_audio_with_sr(args.source)
    input_sr = (
        int(args.input_sr) if args.input_sr and int(args.input_sr) > 0 else source_sr
    )

    block_frame = int(round(args.block_time * input_sr))
    if block_frame <= 0:
        raise ValueError("block_time * input_sr must produce a positive frame count")

    start_payload = build_start_payload(args)
    start_payload["input_sr"] = input_sr

    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=max(30.0, args.http_timeout)
    ) as client:
        resp = await client.post("/realtime/start", json=start_payload)
        resp.raise_for_status()
        print_rvc_theoretical_latency(args, input_sr=input_sr)
        state_resp = await client.get("/realtime/state")
        state_resp.raise_for_status()
        state = state_resp.json()
        output_sr = int(state.get("sample_rate") or input_sr)

    http_rtt_stats = await measure_http_rtt(
        args.base_url, args.http_rtt_count, args.http_timeout
    )
    if http_rtt_stats is not None:
        print("HTTP RTT:")
        print(json.dumps(http_rtt_stats, ensure_ascii=False, indent=2))

    import websockets

    ws_url = (
        args.base_url.replace("http://", "ws://").replace("https://", "wss://")
        + "/realtime/ws"
    )
    output_chunks = []
    event_count = 0
    ws_rtt_stats = None
    rvc_logger = RVCPerformanceLogger(enabled=True)

    def log_chunk(prefix: str, chunk_index: int, sample_count: int, extra: str = ""):
        now = time.strftime("%H:%M:%S")
        suffix = f" {extra}" if extra else ""
        print(
            f"[{now}] {prefix} chunk={chunk_index} samples={sample_count}{suffix}",
            flush=True,
        )

    async def receiver_loop(ws, stop_event: asyncio.Event):
        nonlocal event_count
        recv_chunk_index = 0
        try:
            while not stop_event.is_set():
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if isinstance(message, str):
                    event_count += 1
                    try:
                        payload = json.loads(message)
                        msg_type = payload.get("type", "text")
                        seq = payload.get("sequence_id", "?")
                        sample_count = payload.get("sample_count", "?")
                        infer_time_ms = payload.get("infer_time_ms", "?")
                        if msg_type == "audio_chunk":
                            rvc_logger.handle_audio_meta(payload)
                        print(
                            f"[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}] recv-meta type={msg_type} seq={seq} sample_count={sample_count} infer_ms={infer_time_ms}",
                            flush=True,
                        )
                    except Exception:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] recv-text raw={message}",
                            flush=True,
                        )
                    continue
                chunk = np.frombuffer(message, dtype=np.float32).copy()
                output_chunks.append(chunk)
                log_chunk("recv-audio", recv_chunk_index, int(chunk.shape[0]))
                recv_chunk_index += 1
        except ConnectionClosed:
            return

    async def sender_loop(ws):
        block_duration = block_frame / float(input_sr)
        start_time = time.perf_counter()
        num_blocks = (
            int(np.ceil(len(source_wave) / block_frame)) if len(source_wave) > 0 else 0
        )
        for i in range(num_blocks):
            target_time = start_time + i * block_duration
            now = time.perf_counter()
            if target_time > now:
                await asyncio.sleep(target_time - now)
            start = i * block_frame
            end = min((i + 1) * block_frame, len(source_wave))
            chunk = source_wave[start:end]
            rvc_logger.track_sent_chunk(i, int(chunk.shape[0]))
            await ws.send(chunk.astype(np.float32, copy=False).tobytes())
            lag_ms = (time.perf_counter() - target_time) * 1000.0
            log_chunk("send", i, int(chunk.shape[0]), f"lag_ms={lag_ms:.2f}")
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            if args.ws_rtt_count > 0:
                ws_rtt_stats = await measure_ws_rtt(
                    ws, args.ws_rtt_count, args.ws_rtt_interval
                )
                print("WebSocket RTT:")
                print(json.dumps(ws_rtt_stats, ensure_ascii=False, indent=2))
                print("Recommended runtime parameters:")
                print(
                    json.dumps(
                        recommend_runtime_parameters(
                            ws_rtt_stats["avg_ms"],
                            args.block_time,
                            args.reconnect_delay_ms,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            stop_event = asyncio.Event()
            receiver_task = asyncio.create_task(receiver_loop(ws, stop_event))
            await sender_loop(ws)
            deadline = time.perf_counter() + args.tail_wait_ms / 1000.0
            while time.perf_counter() < deadline:
                await asyncio.sleep(0.05)
            stop_event.set()
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
    finally:
        async with httpx.AsyncClient(
            base_url=args.base_url, timeout=max(30.0, args.http_timeout)
        ) as client:
            try:
                await client.post("/realtime/stop")
            except Exception:
                pass

    output_wave = (
        np.concatenate(output_chunks)
        if output_chunks
        else np.zeros(0, dtype=np.float32)
    )
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sf.write(args.output, output_wave, output_sr)
    print(f"offline_received_chunks={len(output_chunks)}")
    print(f"offline_received_samples={len(output_wave)}")
    print(f"offline_event_count={event_count}")
    print(f"offline_output_sr={output_sr}")
    print(f"offline_output={args.output}")
    rvc_summary = rvc_logger.get_summary()
    if rvc_summary is not None:
        print("offline_rvc_backend:")
        print(json.dumps(rvc_summary, ensure_ascii=False, indent=2))


class DeviceAudioBridge:
    def __init__(
        self,
        samplerate: int,
        block_frame: int,
        input_device,
        output_device,
        input_queue_blocks: int,
        output_queue_blocks: int,
        rtt_probe_count: int,
        rtt_probe_interval: float,
        low_latency_drop_old: bool,
        stream_latency_mode,
        extra_settings,
    ):
        self.samplerate = samplerate
        self.block_frame = block_frame
        self.channels = 1
        self.input_device = input_device
        self.output_device = output_device
        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(1, input_queue_blocks)
        )
        self.output_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(1, output_queue_blocks)
        )
        self.stream: sd.Stream | None = None
        self.stats = RuntimeStats()
        self.started = False
        self.start_time = 0.0
        self.rtt_probe_count = rtt_probe_count
        self.rtt_probe_interval = rtt_probe_interval
        self.low_latency_drop_old = bool(low_latency_drop_old)
        self.stream_latency_mode = stream_latency_mode
        self.extra_settings = extra_settings
        self.send_sequence = 0
        self.rvc_logger = RVCPerformanceLogger(enabled=True)

    def start(self):
        if self.started:
            return
        self.stream = sd.Stream(
            samplerate=self.samplerate,
            blocksize=self.block_frame,
            latency=self.stream_latency_mode,
            dtype="float32",
            channels=self.channels,
            device=(self.input_device, self.output_device),
            extra_settings=self.extra_settings,
            callback=self._audio_callback,
        )
        self.stream.start()
        self.started = True
        self.start_time = time.perf_counter()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
        self.started = False

    def _audio_callback(self, indata, outdata, frames, times, status):
        if status.input_overflow:
            self.stats.input_overflow_count += 1
        if status.input_underflow:
            self.stats.input_underflow_count += 1
        if status.output_overflow:
            self.stats.output_overflow_count += 1
        if status.output_underflow:
            self.stats.output_underflow_count += 1

        mono_in = np.asarray(indata[:, 0], dtype=np.float32).copy()
        if self.low_latency_drop_old:
            dropped_for_latency = 0
            while True:
                try:
                    self.input_queue.get_nowait()
                    dropped_for_latency += 1
                except queue.Empty:
                    break
            if dropped_for_latency > 0:
                self.stats.dropped_input_blocks_for_latency += dropped_for_latency
                self.stats.dropped_input_chunks += dropped_for_latency
            try:
                self.input_queue.put_nowait(mono_in)
            except queue.Full:
                self.stats.dropped_input_chunks += 1
        else:
            try:
                self.input_queue.put_nowait(mono_in)
            except queue.Full:
                self.stats.dropped_input_chunks += 1

        try:
            play = self.output_queue.get_nowait()
        except queue.Empty:
            play = np.zeros(frames, dtype=np.float32)

        if play.shape[0] < frames:
            padded = np.zeros(frames, dtype=np.float32)
            padded[: play.shape[0]] = play
            play = padded
        elif play.shape[0] > frames:
            try:
                self.output_queue.put_nowait(play[frames:].copy())
            except queue.Full:
                self.stats.dropped_output_chunks += 1
            play = play[:frames]

        outdata[:, 0] = play

    async def get_input_block(self, timeout: float = 0.25) -> np.ndarray | None:
        try:
            return await asyncio.to_thread(self.input_queue.get, True, timeout)
        except queue.Empty:
            return None

    async def submit_output_block(self, block: np.ndarray):
        block = np.asarray(block, dtype=np.float32)
        if self.low_latency_drop_old:
            dropped_for_latency = 0
            while True:
                try:
                    self.output_queue.get_nowait()
                    dropped_for_latency += 1
                except queue.Empty:
                    break
            if dropped_for_latency > 0:
                self.stats.dropped_output_blocks_for_latency += dropped_for_latency
                self.stats.dropped_output_chunks += dropped_for_latency
        try:
            self.output_queue.put_nowait(block)
            self.stats.received_chunks += 1
            self.stats.received_samples += int(block.shape[0])
        except queue.Full:
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(block)
                self.stats.dropped_output_chunks += 1
                self.stats.received_chunks += 1
                self.stats.received_samples += int(block.shape[0])
            except queue.Full:
                self.stats.dropped_output_chunks += 1

    def get_summary(self):
        latency_ms = (
            _format_latency_ms(self.stream.latency) if self.stream is not None else None
        )
        summary = {
            "sent_chunks": self.stats.sent_chunks,
            "sent_samples": self.stats.sent_samples,
            "dropped_input_chunks": self.stats.dropped_input_chunks,
            "received_chunks": self.stats.received_chunks,
            "received_samples": self.stats.received_samples,
            "dropped_output_chunks": self.stats.dropped_output_chunks,
            "received_events": self.stats.received_events,
            "reconnect_count": self.stats.reconnect_count,
            "input_overflow_count": self.stats.input_overflow_count,
            "input_underflow_count": self.stats.input_underflow_count,
            "output_overflow_count": self.stats.output_overflow_count,
            "output_underflow_count": self.stats.output_underflow_count,
            "dropped_input_blocks_for_latency": self.stats.dropped_input_blocks_for_latency,
            "dropped_output_blocks_for_latency": self.stats.dropped_output_blocks_for_latency,
            "stream_latency_ms": latency_ms,
            "uptime_sec": (
                time.perf_counter() - self.start_time if self.started else 0.0
            ),
            "pending_input_blocks": self.input_queue.qsize(),
            "pending_output_blocks": self.output_queue.qsize(),
        }
        rvc_summary = self.rvc_logger.get_summary()
        if rvc_summary is not None:
            summary["rvc_backend"] = rvc_summary
        return summary


async def receive_outputs(
    ws, audio_bridge: DeviceAudioBridge, stop_event: asyncio.Event
):
    try:
        while not stop_event.is_set():
            message = await ws.recv()
            if isinstance(message, str):
                audio_bridge.stats.received_events += 1
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue
                msg_type = payload.get("type")
                if msg_type == "audio_chunk":
                    audio_bridge.rvc_logger.handle_audio_meta(payload)
                    continue
                if msg_type in {"pong", "rtt_probe_response"}:
                    continue
                continue
            chunk = np.frombuffer(message, dtype=np.float32).copy()
            await audio_bridge.submit_output_block(chunk)
    except ConnectionClosed:
        return


async def ping_loop(ws, stop_event: asyncio.Event, ping_interval: float):
    while not stop_event.is_set():
        await asyncio.sleep(ping_interval)
        if stop_event.is_set():
            return
        await ws.send(json.dumps({"type": "ping"}))


async def capture_and_send(
    ws, audio_bridge: DeviceAudioBridge, stop_event: asyncio.Event
):
    while not stop_event.is_set():
        block = await audio_bridge.get_input_block(timeout=0.25)
        if block is None:
            continue
        sequence_id = audio_bridge.send_sequence
        audio_bridge.send_sequence += 1
        audio_bridge.rvc_logger.track_sent_chunk(sequence_id, int(block.shape[0]))
        await ws.send(block.astype(np.float32, copy=False).tobytes())
        audio_bridge.stats.sent_chunks += 1
        audio_bridge.stats.sent_samples += int(block.shape[0])


async def stream_session(
    ws_url: str,
    audio_bridge: DeviceAudioBridge,
    reconnect_delay_ms: float,
    ping_interval: float,
    stop_event: asyncio.Event,
):
    import websockets

    first_connect = True

    while not stop_event.is_set():
        try:
            async with websockets.connect(ws_url, max_size=None) as ws:
                if first_connect:
                    ws_rtt_stats = await measure_ws_rtt(
                        ws,
                        audio_bridge.rtt_probe_count,
                        audio_bridge.rtt_probe_interval,
                    )
                    if ws_rtt_stats is not None:
                        print("WebSocket RTT:")
                        print(json.dumps(ws_rtt_stats, ensure_ascii=False, indent=2))
                        recommended = recommend_runtime_parameters(
                            ws_rtt_stats["avg_ms"],
                            audio_bridge.block_frame / audio_bridge.samplerate,
                            reconnect_delay_ms,
                        )
                        print("Recommended runtime parameters:")
                        print(json.dumps(recommended, ensure_ascii=False, indent=2))
                    first_connect = False
                receiver_task = asyncio.create_task(
                    receive_outputs(ws, audio_bridge, stop_event)
                )
                sender_task = asyncio.create_task(
                    capture_and_send(ws, audio_bridge, stop_event)
                )
                pinger_task = asyncio.create_task(
                    ping_loop(ws, stop_event, ping_interval)
                )
                done, pending = await asyncio.wait(
                    {receiver_task, sender_task, pinger_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                for task in done:
                    exc = None
                    try:
                        exc = task.exception()
                    except asyncio.CancelledError:
                        pass
                    if exc is not None and not isinstance(exc, ConnectionClosed):
                        raise exc
        except ConnectionClosed:
            if stop_event.is_set():
                break
        except Exception:
            if stop_event.is_set():
                break
            audio_bridge.stats.reconnect_count += 1
            await asyncio.sleep(reconnect_delay_ms / 1000.0)
            continue

        if stop_event.is_set():
            break
        audio_bridge.stats.reconnect_count += 1
        await asyncio.sleep(reconnect_delay_ms / 1000.0)


def start_stop_watcher(stop_event: asyncio.Event):
    def worker():
        try:
            input("按回车停止会话...\n")
        except EOFError:
            return
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_signal_stop(stop_event))
        loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


async def _signal_stop(stop_event: asyncio.Event):
    stop_event.set()


async def main_async(args):
    block_frame = int(round(args.block_time * args.input_sr))
    if block_frame <= 0:
        raise ValueError("block_time * input_sr must produce a positive frame count")

    start_payload = build_start_payload(args)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
        resp = await client.post("/realtime/start", json=start_payload)
        resp.raise_for_status()
    print_rvc_theoretical_latency(args)

    http_rtt_stats = await measure_http_rtt(
        args.base_url, args.http_rtt_count, args.http_timeout
    )
    if http_rtt_stats is not None:
        print("HTTP RTT:")
        print(json.dumps(http_rtt_stats, ensure_ascii=False, indent=2))

    audio_bridge = DeviceAudioBridge(
        samplerate=args.input_sr,
        block_frame=block_frame,
        input_device=parse_device_argument(args.input_device),
        output_device=parse_device_argument(args.output_device),
        input_queue_blocks=args.client_input_queue_blocks,
        output_queue_blocks=args.client_output_queue_blocks,
        rtt_probe_count=args.ws_rtt_count,
        rtt_probe_interval=args.ws_rtt_interval,
        low_latency_drop_old=args.low_latency_drop_old,
        stream_latency_mode=args.stream_latency,
        extra_settings=create_sounddevice_extra_settings(
            args.low_latency_wasapi, args.wasapi_exclusive
        ),
    )

    stop_event = asyncio.Event()
    ws_url = (
        args.base_url.replace("http://", "ws://").replace("https://", "wss://")
        + "/realtime/ws"
    )
    audio_bridge.start()
    watcher = start_stop_watcher(stop_event)

    try:
        await stream_session(
            ws_url=ws_url,
            audio_bridge=audio_bridge,
            reconnect_delay_ms=args.reconnect_delay_ms,
            ping_interval=args.ping_interval,
            stop_event=stop_event,
        )
    finally:
        audio_bridge.stop()
        async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
            try:
                await client.post("/realtime/stop")
            except Exception:
                pass

    print(json.dumps(audio_bridge.get_summary(), ensure_ascii=False, indent=2))
    if watcher.is_alive():
        watcher.join(timeout=0.1)


def main():
    parser = argparse.ArgumentParser(
        description="RVC realtime CLI client for realtime_v2 FastAPI server"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="device",
        choices=["device", "rtt", "offline"],
        help="运行模式：设备实时 / 纯 RTT 测试 / 基于实时 API 的离线推理",
    )
    parser.add_argument(
        "--list-hostapis", action="store_true", help="列出所有可用的音频 API 分类并退出"
    )
    parser.add_argument(
        "--list-in-devices",
        action="store_true",
        help="列出可用作输入（麦克风/采集，即 max_input_ch > 0）的音频设备并退出",
    )
    parser.add_argument(
        "--list-out-devices",
        action="store_true",
        help="列出可用作输出（扬声器/播放，即 max_output_ch > 0）的音频设备并退出",
    )
    parser.add_argument(
        "--hostapi-filter",
        type=str,
        default=None,
        help="按音频 API 名称（模糊匹配）过滤 --list-in-devices / --list-out-devices 的输出，例如 'WASAPI' 或 'ALSA'",
    )
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:6243")
    parser.add_argument("--pth", type=str, required=False)
    parser.add_argument("--index", type=str, default="")
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--formant", type=float, default=0.0)
    parser.add_argument("--block-time", type=float, default=0.25)
    parser.add_argument("--crossfade-time", type=float, default=0.05)
    parser.add_argument("--extra-time", type=float, default=2.5)
    parser.add_argument(
        "--input-sr",
        type=int,
        default=0,
        help="device 模式下必填；offline 模式下为 0 时自动读取 source 文件采样率",
    )
    parser.add_argument("--input-queue-size", type=int, default=8)
    parser.add_argument("--input-ttl-ms", type=float, default=500.0)
    parser.add_argument("--output-ttl-ms", type=float, default=500.0)
    parser.add_argument("--reconnect-ttl-ms", type=float, default=30000.0)
    parser.add_argument("--use-phase-vocoder", action="store_true")
    parser.add_argument(
        "--input-device", type=str, default=None, help="输入设备 index 或名称"
    )
    parser.add_argument(
        "--output-device", type=str, default=None, help="输出设备 index 或名称"
    )
    parser.add_argument("--client-input-queue-blocks", type=int, default=8)
    parser.add_argument("--client-output-queue-blocks", type=int, default=8)
    parser.add_argument(
        "--low-latency-drop-old",
        action="store_true",
        help="低延迟模式：输入/输出队列只保留最新块，主动丢弃旧块",
    )
    parser.add_argument(
        "--stream-latency",
        type=str,
        default="low",
        help="sounddevice latency 参数，常用值：low / high / 数字秒",
    )
    parser.add_argument(
        "--low-latency-wasapi",
        action="store_true",
        help="Windows/WASAPI 下启用低延迟 extra_settings",
    )
    parser.add_argument(
        "--wasapi-exclusive",
        action="store_true",
        help="WASAPI 独占模式，通常可进一步降低延迟",
    )
    parser.add_argument("--reconnect-delay-ms", type=float, default=800.0)
    parser.add_argument("--ping-interval", type=float, default=5.0)
    parser.add_argument(
        "--http-rtt-count",
        type=int,
        default=5,
        help="启动后先通过 HTTP RTT 端点测量往返延迟次数",
    )
    parser.add_argument(
        "--http-timeout", type=float, default=10.0, help="HTTP RTT 测试超时秒数"
    )
    parser.add_argument(
        "--ws-rtt-count",
        type=int,
        default=5,
        help="首次连上 WS 后发送 RTT probe 的次数",
    )
    parser.add_argument(
        "--ws-rtt-interval", type=float, default=0.1, help="WS RTT probe 间隔秒数"
    )
    parser.add_argument(
        "--source", type=str, default="", help="offline 模式输入 wav/flac 文件"
    )
    parser.add_argument(
        "--output", type=str, default="", help="offline 模式输出 wav 文件"
    )
    parser.add_argument(
        "--send-interval-ms",
        type=float,
        default=0.0,
        help="offline 模式块发送间隔，0 表示尽快发送",
    )
    parser.add_argument(
        "--tail-wait-ms",
        type=float,
        default=3000.0,
        help="offline 模式 flush 后等待尾音的时长",
    )
    parser.add_argument(
        "--auto-start-for-rtt",
        action="store_true",
        help="rtt 模式下自动 start/stop 一个实时 session，以便测试 WS RTT",
    )
    parser.add_argument(
        "--silero-vad",
        action="store_true",
        help="启用Silero VAD"
    )
    args = parser.parse_args()

    if args.stream_latency not in ("low", "high"):
        try:
            args.stream_latency = float(args.stream_latency)
        except ValueError as exc:
            raise SystemExit(
                "--stream-latency must be 'low', 'high', or a float number of seconds"
            ) from exc

    if args.list_hostapis:
        list_hostapis()
        return
    if args.list_in_devices:
        list_audio_devices(hostapi_filter=args.hostapi_filter, direction="input")
        return
    if args.list_out_devices:
        list_audio_devices(hostapi_filter=args.hostapi_filter, direction="output")
        return
    if args.mode == "rtt":
        if not args.pth and args.auto_start_for_rtt:
            raise SystemExit("--pth is required when --auto-start-for-rtt is used")
        asyncio.run(run_rtt_test_mode(args))
        return
    if not args.pth:
        raise SystemExit("--pth is required unless --list-in-devices / --list-out-devices / --list-hostapis is used")
    if args.mode == "offline":
        asyncio.run(run_offline_mode(args))
        return
    if args.input_sr <= 0:
        raise SystemExit("device 模式需要提供正整数 --input-sr")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
