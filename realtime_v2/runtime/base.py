from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from realtime_v2.engine import RealtimeRVCEngine


@dataclass
class AudioStream:
    name: str
    base_sr: int
    base_samples: np.ndarray
    views: dict[int, np.ndarray]

    def get_view(
        self,
        sr: int,
        *,
        resampler: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> np.ndarray:
        if sr not in self.views:
            if sr == self.base_sr:
                self.views[sr] = self.base_samples
            else:
                self.views[sr] = resampler(self.base_samples, self.base_sr, sr)
        return self.views[sr]


@dataclass
class InputChunk:
    sequence_id: int
    streams: dict[str, AudioStream]
    features: dict[str, Any]
    created_at: float
    expire_at: float

    def set_stream(self, name: str, samples: np.ndarray, sr: int):
        self.streams[name] = AudioStream(
            name=name,
            base_sr=sr,
            base_samples=samples.astype(np.float32, copy=False),
            views={},
        )

    def has_stream(self, name: str) -> bool:
        return name in self.streams

    def get_stream(self, name: str) -> AudioStream:
        if name not in self.streams:
            raise KeyError(f"stream not found: {name}")
        return self.streams[name]

    def get_stream_view(
        self,
        name: str,
        sr: int,
        *,
        resampler: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> np.ndarray:
        return self.get_stream(name).get_view(sr, resampler=resampler)

    def get_stream_base_samples(self, name: str) -> np.ndarray:
        return self.get_stream(name).base_samples

    def set_feature(self, name: str, value: Any):
        self.features[name] = value

    def get_feature(self, name: str, default: Any = None):
        return self.features.get(name, default)


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
    engine_input_stream: str
    block_16k: np.ndarray | None
    output_override: np.ndarray | None = None


class FrameAnalyzer(Protocol):
    async def analyze(self, chunk: InputChunk, context: AnalysisContext): ...


class FramePolicy(Protocol):
    def decide(self, chunk: InputChunk, context: ExecutionContext) -> GateDecision: ...