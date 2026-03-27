from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import FrameAnalyzer, FramePolicy
from .executor import FrameExecutor
from .policies import PassthroughPolicy
from .vad import SileroVAD, SileroVADAnalyzer, SpeechGatePolicy

if TYPE_CHECKING:
    from realtime_v2.server import StartRequest
    from realtime_v2.model_adapter import RVCRealtimeAdapter


@dataclass
class RuntimePipeline:
    analyzers: list[FrameAnalyzer]
    policy: FramePolicy
    executor: FrameExecutor
    vad: SileroVAD | None = None


async def build_runtime(req: "StartRequest", adapter: "RVCRealtimeAdapter") -> RuntimePipeline:
    analyzers: list[FrameAnalyzer] = []
    policy: FramePolicy = PassthroughPolicy()
    vad: SileroVAD | None = None

    if req.enable_vad:
        device_str = str(adapter.device) if adapter is not None else "cpu"
        vad = SileroVAD(threshold=req.vad_threshold, device=device_str)
        await __import__("asyncio").to_thread(vad.warmup)
        analyzers.append(SileroVADAnalyzer(vad))
        policy = SpeechGatePolicy()

    return RuntimePipeline(
        analyzers=analyzers,
        policy=policy,
        executor=FrameExecutor(),
        vad=vad,
    )