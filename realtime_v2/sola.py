import sys

import torch
import torch.nn.functional as F


def phase_vocoder(a: torch.Tensor, b: torch.Tensor, fade_out: torch.Tensor, fade_in: torch.Tensor):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * torch.pi * torch.floor(deltaphase / (2 * torch.pi) + 0.5)
    w = 2 * torch.pi * torch.arange(n // 2 + 1, device=a.device, dtype=a.dtype) + deltaphase
    t = torch.arange(n, device=a.device, dtype=a.dtype).unsqueeze(-1) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), dim=-1) * window / n
    )
    return result


class SOLAAligner:
    def __init__(self, sola_buffer_frame: int, sola_search_frame: int, device: torch.device, use_phase_vocoder: bool = False):
        self.sola_buffer_frame = sola_buffer_frame
        self.sola_search_frame = sola_search_frame
        self.device = device
        self.use_phase_vocoder = use_phase_vocoder
        self.buffer = torch.zeros(sola_buffer_frame, device=device, dtype=torch.float32)
        self.fade_in_window = (
            torch.sin(
                0.5
                * torch.pi
                * torch.linspace(0.0, 1.0, steps=sola_buffer_frame, device=device, dtype=torch.float32)
            )
            ** 2
        )
        self.fade_out_window = 1.0 - self.fade_in_window

    def align_and_blend(self, infer_wav: torch.Tensor, block_frame: int):
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            )
            + 1e-8
        )
        corr = cor_nom[0, 0] / cor_den[0, 0]
        if sys.platform == "darwin":
            _, sola_offset = torch.max(corr, dim=0)
            sola_offset = int(sola_offset.item())
        else:
            sola_offset = int(torch.argmax(corr, dim=0).item())

        aligned = infer_wav[sola_offset:]
        if self.use_phase_vocoder:
            aligned[: self.sola_buffer_frame] = phase_vocoder(
                self.buffer,
                aligned[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        else:
            aligned[: self.sola_buffer_frame] *= self.fade_in_window
            aligned[: self.sola_buffer_frame] += self.buffer * self.fade_out_window
        self.buffer[:] = aligned[block_frame : block_frame + self.sola_buffer_frame]
        return aligned, sola_offset
