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
        if decision.engine_input_stream == "__silence__":
            engine_input = chunk.get_feature("silence_input")
            if engine_input is None:
                base_stream = "clean" if chunk.has_stream("clean") else "input"
                base = chunk.get_stream_base_samples(base_stream)
                engine_input = __import__("numpy").zeros(base.shape[0], dtype=base.dtype)
        else:
            engine_input = chunk.get_stream_base_samples(decision.engine_input_stream)
        result = await asyncio.to_thread(
            context.engine.process_block,
            engine_input,
            silence_16k=decision.block_16k,
        )
        output_wave = decision.output_override
        if output_wave is None:
            output_wave = result.output_wave[: engine_input.shape[0]]
        return output_wave, result.infer_time_ms