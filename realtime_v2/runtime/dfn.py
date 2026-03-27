from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from .base import AnalysisContext, FrameAnalyzer, InputChunk

from dfnstream_py import DeepFilterNetStreaming, DeepFilterNetStreamingONNX


@dataclass
class DFNConfig:
    backend: str = "native"
    model_path: str = ""
    atten_lim: float | None = None
    post_filter_beta: float = 0.0
    compensate_delay: bool = False
    input_stream: str = "input"
    output_stream: str = "clean"


class DFNProcessor:
    def __init__(self, config: DFNConfig):
        self.config = config
        processor_cls = (
            DeepFilterNetStreamingONNX
            if config.backend.lower() == "onnx"
            else DeepFilterNetStreaming
        )
        kwargs = {
            "model_path": config.model_path or None,
            "atten_lim": config.atten_lim,
            "post_filter_beta": config.post_filter_beta,
            "compensate_delay": config.compensate_delay,
        }
        self.processor = processor_cls(**kwargs)
        self.output_buffer = np.array([], dtype=np.float32)

    @property
    def sample_rate(self) -> int:
        return int(self.processor.sample_rate)

    def close(self):
        self.processor.close()
        self.output_buffer = np.array([], dtype=np.float32)

    def process(
        self,
        samples: np.ndarray,
        input_sr: int,
        *,
        resampler,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_len = samples.shape[0]
        samples_48k = resampler(samples, input_sr, self.sample_rate)
        processed_48k = self.processor.process_chunk(samples_48k)
        if processed_48k.size > 0:
            self.output_buffer = np.concatenate([self.output_buffer, processed_48k])

        needed_48k = int(round(target_len * self.sample_rate / input_sr))
        if self.output_buffer.shape[0] >= needed_48k:
            aligned_48k = self.output_buffer[:needed_48k]
            self.output_buffer = self.output_buffer[needed_48k:]
        else:
            deficit = needed_48k - self.output_buffer.shape[0]
            fallback = samples_48k[-deficit:] if deficit > 0 else np.array([], dtype=np.float32)
            aligned_48k = np.concatenate([self.output_buffer, fallback])
            self.output_buffer = np.array([], dtype=np.float32)

        aligned_in_sr = resampler(aligned_48k, self.sample_rate, input_sr)
        if aligned_in_sr.shape[0] != target_len:
            if aligned_in_sr.shape[0] > target_len:
                aligned_in_sr = aligned_in_sr[:target_len]
            else:
                pad = target_len - aligned_in_sr.shape[0]
                aligned_in_sr = np.pad(aligned_in_sr, (0, pad))
        return aligned_in_sr.astype(np.float32, copy=False), aligned_48k.astype(np.float32, copy=False)


class DFNAnalyzer(FrameAnalyzer):
    def __init__(self, processor: DFNProcessor, input_stream: str = "input", output_stream: str = "clean"):
        self.processor = processor
        self.input_stream = input_stream
        self.output_stream = output_stream

    async def analyze(self, chunk: InputChunk, context: AnalysisContext):
        stream = chunk.get_stream(self.input_stream)
        clean_samples, clean_48k = await asyncio.to_thread(
            self.processor.process,
            stream.base_samples,
            stream.base_sr,
            resampler=context.resampler,
        )
        chunk.set_stream(self.output_stream, clean_samples, stream.base_sr)
        chunk.set_feature(f"{self.output_stream}_48k", clean_48k)
        input_rms = (
            float(np.sqrt(np.mean(np.square(stream.base_samples, dtype=np.float32))))
            if stream.base_samples.size > 0
            else 0.0
        )
        output_rms = (
            float(np.sqrt(np.mean(np.square(clean_samples, dtype=np.float32))))
            if clean_samples.size > 0
            else 0.0
        )
        residual = stream.base_samples - clean_samples
        residual_rms = (
            float(np.sqrt(np.mean(np.square(residual, dtype=np.float32))))
            if residual.size > 0
            else 0.0
        )
        chunk.set_feature("dfn_enabled", True)
        chunk.set_feature("dfn_input_rms", input_rms)
        chunk.set_feature("dfn_output_rms", output_rms)
        chunk.set_feature("dfn_residual_rms", residual_rms)
        chunk.set_feature(
            "dfn_rms_ratio",
            float(output_rms / input_rms) if input_rms > 1e-8 else None,
        )