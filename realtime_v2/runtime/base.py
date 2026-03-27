from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from realtime_v2.engine import RealtimeRVCEngine


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


class FrameAnalyzer(Protocol):
    async def analyze(self, chunk: InputChunk, context: AnalysisContext): ...


class FramePolicy(Protocol):
    def decide(self, chunk: InputChunk, context: ExecutionContext) -> GateDecision: ...