from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os

import faiss
import fairseq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample
from fairseq.data.dictionary import Dictionary

from configs.config import Config
from infer.lib import jit
from infer.lib.jit.get_synthesizer import get_synthesizer


@dataclass
class RVCModelConfig:
    pth_path: str
    index_path: str = ""
    index_rate: float = 0.0
    pitch_shift: int = 0
    formant_shift: float = 0.0


class RVCRealtimeAdapter:
    def __init__(self, model_config: RVCModelConfig, config: Config, last_adapter: "RVCRealtimeAdapter | None" = None):
        self.model_config = model_config
        self.config = config
        self.device = config.device if isinstance(config.device, torch.device) else torch.device(config.device)
        self.is_half = config.is_half
        self.use_jit = config.use_jit
        self.pitch_shift = model_config.pitch_shift
        self.formant_shift = model_config.formant_shift
        self.index_path = model_config.index_path
        self.index_rate = model_config.index_rate
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.cache_pitch = torch.zeros(1024, device=self.device, dtype=torch.long)
        self.cache_pitchf = torch.zeros(1024, device=self.device, dtype=torch.float32)
        self.resample_kernel: dict[int, Resample] = {}
        self.input_sr = 16000

        try:
            torch.serialization.add_safe_globals([Dictionary])
        except AttributeError:
            pass

        if last_adapter is None:
            models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                "assets/hubert/hubert_base.pt"
            ], suffix="")
            self.hubert_model = models[0].to(self.device)
            self.hubert_model = self.hubert_model.half() if self.is_half else self.hubert_model.float()
            self.hubert_model.eval()
        else:
            self.hubert_model = last_adapter.hubert_model

        self.net_g = None
        self._load_synthesizer(last_adapter)
        if last_adapter is not None and hasattr(last_adapter, "model_rmvpe"):
            self.model_rmvpe = last_adapter.model_rmvpe

        if self.index_rate != 0 and self.index_path:
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

    def _load_default_model(self):
        self.net_g, cpt = get_synthesizer(self.model_config.pth_path, self.device)
        self.tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = cpt.get("f0", 1)
        self.version = cpt.get("version", "v1")
        self.net_g = self.net_g.half() if self.is_half else self.net_g.float()

    def _load_jit_model(self):
        jit_pth_path = self.model_config.pth_path.rstrip(".pth")
        jit_pth_path += ".half.jit" if self.is_half else ".jit"
        reload = False
        device = self.device
        if str(device) == "cuda":
            device = torch.device("cuda:0")
        if os.path.exists(jit_pth_path):
            cpt = jit.load(jit_pth_path)
            if cpt["device"] != str(device):
                reload = True
        else:
            reload = True
        if reload:
            cpt = jit.synthesizer_jit_export(
                self.model_config.pth_path,
                "script",
                None,
                device=device,
                is_half=self.is_half,
            )
        self.tgt_sr = cpt["config"][-1]
        self.if_f0 = cpt.get("f0", 1)
        self.version = cpt.get("version", "v1")
        self.net_g = torch.jit.load(BytesIO(cpt["model"]), map_location=device)
        self.net_g.infer = self.net_g.forward
        self.net_g.eval().to(device)

    def _load_synthesizer(self, last_adapter: "RVCRealtimeAdapter | None"):
        if last_adapter is not None and last_adapter.model_config.pth_path == self.model_config.pth_path and last_adapter.use_jit == self.use_jit:
            self.tgt_sr = last_adapter.tgt_sr
            self.if_f0 = last_adapter.if_f0
            self.version = last_adapter.version
            self.net_g = last_adapter.net_g
            return
        if self.use_jit and not self.config.dml and not (self.is_half and "cpu" in str(self.device)):
            self._load_jit_model()
        else:
            self._load_default_model()

    def change_pitch_shift(self, value: int):
        self.pitch_shift = int(value)

    def change_formant_shift(self, value: float):
        self.formant_shift = float(value)

    def change_index_rate(self, value: float):
        value = float(value)
        if value != 0 and self.index_rate == 0 and self.index_path:
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
        self.index_rate = value

    def _get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0_rmvpe(self, x: torch.Tensor, f0_up_key: float):
        if not hasattr(self, "model_rmvpe"):
            from infer.lib.rmvpe import RMVPE

            self.model_rmvpe = RMVPE(
                "assets/rmvpe/rmvpe.pt",
                is_half=self.is_half,
                device=self.device,
                use_jit=self.config.use_jit,
            )
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)
        return self._get_f0_post(f0)

    def infer_window(self, input_wav_16k: torch.Tensor, block_frame_16k: int, skip_head: int, return_length: int):
        with torch.no_grad():
            feats = input_wav_16k.half().view(1, -1) if self.is_half else input_wav_16k.float().view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            inputs = {
                "source": feats,
                "padding_mask": padding_mask,
                "output_layer": 9 if self.version == "v1" else 12,
            }
            logits = self.hubert_model.extract_features(**inputs)
            feats = self.hubert_model.final_proj(logits[0]) if self.version == "v1" else logits[0]
            feats = torch.cat((feats, feats[:, -1:, :]), 1)

        if hasattr(self, "index") and self.index_rate != 0:
            npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
            score, ix = self.index.search(npy, k=8)
            if (ix >= 0).all():
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                if self.is_half:
                    npy = npy.astype("float16")
                feats[0][skip_head // 2 :] = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * self.index_rate
                    + (1 - self.index_rate) * feats[0][skip_head // 2 :]
                )

        p_len = input_wav_16k.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))

        cache_pitch = None
        cache_pitchf = None
        if self.if_f0 == 1:
            f0_extractor_frame = block_frame_16k + 800
            f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            pitch, pitchf = self.get_f0_rmvpe(
                input_wav_16k[-f0_extractor_frame:],
                self.pitch_shift - self.formant_shift,
            )
            shift = block_frame_16k // 160
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = self.cache_pitch[None, -p_len:]
            cache_pitchf = self.cache_pitchf[None, -p_len:] * return_length2 / return_length

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        p_len_t = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head_t = torch.LongTensor([skip_head])
        return_length2_t = torch.LongTensor([return_length2])
        return_length_t = torch.LongTensor([return_length])

        with torch.no_grad():
            if self.if_f0 == 1:
                infered_audio, _, _ = self.net_g.infer(
                    feats,
                    p_len_t,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                    skip_head_t,
                    return_length_t,
                    return_length2_t,
                )
            else:
                infered_audio, _, _ = self.net_g.infer(
                    feats, p_len_t, sid, skip_head_t, return_length_t, return_length2_t
                )
        infered_audio = infered_audio.squeeze(1).float()
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                self.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res,
                    new_freq=self.tgt_sr // 100,
                    dtype=torch.float32,
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](infered_audio[:, : return_length * upp_res])
        return infered_audio.squeeze()
