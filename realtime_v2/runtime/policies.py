from __future__ import annotations

from .base import ExecutionContext, FramePolicy, GateDecision, InputChunk


class PassthroughPolicy(FramePolicy):
    def decide(self, chunk: InputChunk, context: ExecutionContext) -> GateDecision:
        block_16k = chunk.get_feature("analysis_view_16k")
        return GateDecision(
            action="run_rvc",
            engine_input=chunk.master_samples,
            block_16k=block_16k,
        )