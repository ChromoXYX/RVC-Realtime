from __future__ import annotations

import asyncio

from .base import ExecutionContext, GateDecision, InputChunk


class FrameExecutor:
    async def execute(
        self,
        chunk: InputChunk,
        decision: GateDecision,
        context: ExecutionContext,
    ):
        result = await asyncio.to_thread(
            context.engine.process_block,
            decision.engine_input,
            silence_16k=decision.block_16k,
        )
        output_wave = decision.output_override
        if output_wave is None:
            output_wave = result.output_wave[: chunk.master_samples.shape[0]]
        return output_wave, result.infer_time_ms