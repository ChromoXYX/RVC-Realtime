from __future__ import annotations

from dataclasses import dataclass
import time

import librosa
import numpy as np
import torch

from .model_adapter import RVCRealtimeAdapter
from .sola import SOLAAligner


@dataclass
class RealtimeRVCConfig:
    block_time: float = 0.25
    crossfade_time: float = 0.05
    extra_time: float = 2.5
    use_phase_vocoder: bool = False

    def validate(self):
        if self.block_time <= 0:
            raise ValueError("block_time 必须大于 0")
        if self.crossfade_time <= 0:
            raise ValueError("crossfade_time 必须大于 0")
        if self.extra_time < 0:
            raise ValueError("extra_time 不能小于 0")


@dataclass
class BlockResult:
    block_index: int
    output_wave: np.ndarray
    infer_time_ms: float
    sola_offset: int
    input_samples: int


class RealtimeRVCEngine:
    def __init__(self, adapter: RVCRealtimeAdapter, config: RealtimeRVCConfig, input_sr: int | None = None):
        self.adapter = adapter
        self.device = adapter.device
        self.config = config
        self.config.validate()
        self.input_sr = input_sr or adapter.tgt_sr
        self.initialized = False
        self.block_index = 0

    def _seconds_to_aligned_samples(self, seconds: float):
        zc = self.input_sr // 100
        return int(np.round(seconds * self.input_sr / zc)) * zc

    def initialize_stream(self):
        self.zc = self.input_sr // 100
        self.block_frame = self._seconds_to_aligned_samples(self.config.block_time)
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = self._seconds_to_aligned_samples(self.config.crossfade_time)
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = self._seconds_to_aligned_samples(self.config.extra_time)
        total_input_len = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(total_input_len, device=self.device, dtype=torch.float32)
        self.input_wav_16k = torch.zeros(160 * total_input_len // self.zc, device=self.device, dtype=torch.float32)
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.sola = SOLAAligner(
            self.sola_buffer_frame,
            self.sola_search_frame,
            self.device,
            use_phase_vocoder=self.config.use_phase_vocoder,
        )
        self.initialized = True
        self.block_index = 0

    def get_algorithmic_delay_samples(self):
        return self.crossfade_frame + self.sola_search_frame

    def _update_buffers(self, block_wave: np.ndarray):
        if len(block_wave) != self.block_frame:
            padded = np.zeros(self.block_frame, dtype=np.float32)
            padded[: len(block_wave)] = block_wave
            block_wave = padded

        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-self.block_frame :] = torch.from_numpy(block_wave).to(self.device)

        recent = self.input_wav[-self.block_frame - 2 * self.zc :].detach().cpu().numpy()
        recent_16k = librosa.resample(recent, orig_sr=self.input_sr, target_sr=16000)[160:]
        expected = 160 * (self.block_frame // self.zc + 1)
        if len(recent_16k) != expected:
            if len(recent_16k) < expected:
                recent_16k = np.pad(recent_16k, (0, expected - len(recent_16k)))
            else:
                recent_16k = recent_16k[:expected]
        self.input_wav_16k[:-self.block_frame_16k] = self.input_wav_16k[self.block_frame_16k :].clone()
        self.input_wav_16k[-expected:] = torch.from_numpy(recent_16k).to(self.device)

    def _ensure_minimum_length(self, infer_wav: torch.Tensor):
        required = self.sola_buffer_frame + self.sola_search_frame + self.block_frame
        if infer_wav.numel() >= required:
            return infer_wav
        padded = torch.zeros(required, device=infer_wav.device, dtype=infer_wav.dtype)
        if infer_wav.numel() > 0:
            padded[: infer_wav.numel()] = infer_wav
        return padded

    def process_block(self, block_wave: np.ndarray):
        if not self.initialized:
            self.initialize_stream()

        input_samples = len(block_wave)
        self._update_buffers(block_wave.astype(np.float32))

        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            infer_wav = self.adapter.infer_window(
                self.input_wav_16k,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
            )
            end.record()
            torch.cuda.synchronize()
            infer_time_ms = float(start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            infer_wav = self.adapter.infer_window(
                self.input_wav_16k,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
            )
            infer_time_ms = (time.perf_counter() - start_time) * 1000.0

        if self.input_sr != self.adapter.tgt_sr:
            infer_wav = librosa.resample(
                infer_wav.detach().cpu().numpy(),
                orig_sr=self.adapter.tgt_sr,
                target_sr=self.input_sr,
            )
            infer_wav = torch.from_numpy(infer_wav).to(self.device)

        infer_wav = self._ensure_minimum_length(infer_wav)
        aligned, sola_offset = self.sola.align_and_blend(infer_wav, self.block_frame)
        out_block = aligned[: self.block_frame].detach().cpu().numpy()
        result = BlockResult(
            block_index=self.block_index,
            output_wave=out_block,
            infer_time_ms=infer_time_ms,
            sola_offset=sola_offset,
            input_samples=input_samples,
        )
        self.block_index += 1
        return result
