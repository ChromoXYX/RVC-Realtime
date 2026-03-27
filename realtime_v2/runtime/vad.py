from __future__ import annotations

import asyncio

import numpy as np
import torch

from .base import AnalysisContext, ExecutionContext, FrameAnalyzer, FramePolicy, GateDecision, InputChunk


class SileroVAD:
    _FRAME_SIZE = 512

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
        self._get_speech_timestamps = utils[0]

    def reset(self):
        if self._model is not None:
            self._model.reset_states()

    def warmup(self):
        self._ensure_loaded()
        dummy = np.zeros(self._FRAME_SIZE, dtype=np.float32)
        self.is_speech(dummy)
        self.reset()

    def is_speech(self, samples_16k: np.ndarray) -> tuple[bool, float]:
        self._ensure_loaded()
        audio = samples_16k.astype(np.float32)
        remainder = len(audio) % self._FRAME_SIZE
        if remainder != 0:
            audio = np.pad(audio, (0, self._FRAME_SIZE - remainder))
        max_prob = 0.0
        with torch.no_grad():
            for i in range(0, len(audio), self._FRAME_SIZE):
                frame = torch.from_numpy(audio[i : i + self._FRAME_SIZE]).to(self.device)
                p = float(self._model(frame, 16000).item())
                if p > max_prob:
                    max_prob = p
        return max_prob >= self.threshold, max_prob


class SileroVADAnalyzer(FrameAnalyzer):
    def __init__(self, vad: SileroVAD):
        self.vad = vad

    async def analyze(self, chunk: InputChunk, context: AnalysisContext):
        stream_name = "clean" if chunk.has_stream("clean") else "input"
        chunk_16k = chunk.get_stream_view(stream_name, 16000, resampler=context.resampler)
        is_speech, vad_prob = await asyncio.to_thread(self.vad.is_speech, chunk_16k)
        chunk.set_feature("analysis_view_16k", chunk_16k)
        chunk.set_feature("is_speech", is_speech)
        chunk.set_feature("vad_prob", vad_prob)


class SpeechGatePolicy(FramePolicy):
    def decide(self, chunk: InputChunk, context: ExecutionContext) -> GateDecision:
        block_16k = chunk.get_feature("analysis_view_16k")
        stream_name = "clean" if chunk.has_stream("clean") else "input"
        engine_input = chunk.get_stream_base_samples(stream_name)
        if chunk.get_feature("is_speech") is False:
            silence_input = np.zeros(engine_input.shape[0], dtype=np.float32)
            silence_output = np.zeros(engine_input.shape[0], dtype=np.float32)
            return GateDecision(
                action="push_silence",
                engine_input_stream="__silence__",
                block_16k=block_16k,
                output_override=silence_output,
            )
        return GateDecision(
            action="run_rvc",
            engine_input_stream=stream_name,
            block_16k=block_16k,
        )