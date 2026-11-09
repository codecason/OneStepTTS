"""WavTTS MeanFlow interface scaffold, NOT a runnable training implementation.

Time: noise=1 -> clean=0. Waveforms/velocities: [B, N]. Scalar conditions: [B].
Core call chains are written out; unfinished helpers fail explicitly.
See TODO.md for pseudocode, contracts, and integration tasks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PMFOptions:
    """Provisional settings, not experimentally validated pMF audio hyperparameters."""

    latents_scale: float = 9.0
    noise_scale: float = 1.0
    t_eps: float = 0.02
    P_mean: float = 0.8  # Sign-flipped WavTTS mean under t_pmf = 1 - t_wavtts.
    P_std: float = 0.8
    data_proportion: float = 0.5
    cfg_max: float = 7.0
    cfg_beta: float = 1.0
    joint_cond_drop_prob: float = 0.1
    norm_p: float = 1.0
    norm_eps: float = 0.01
    sample_rate: int = 16000
    wav_frame_len: int = 160


@dataclass(frozen=True)
class WaveCondition:
    """Fixed throughout a JVP; audio is scaled, text is already tokenized.

    audio [B,N]: reference samples, zero in the prediction region and padding.
    text [B,L]: token IDs; lengths [B]: valid waveform lengths in samples.
    valid_mask [B,N]: excludes padding; prediction_mask [B,N]: loss region.
    drop_mask [B]: jointly drops audio/text for selected examples.
    """

    audio: Tensor
    text: Tensor
    lengths: Tensor
    valid_mask: Tensor
    prediction_mask: Tensor
    drop_mask: Tensor


@dataclass(frozen=True)
class FlowSchedule:
    """Every field is [B]; 0 <= r <= t <= 1, fm_mask is boolean."""

    t: Tensor
    r: Tensor
    fm_mask: Tensor
    omega: Tensor
    t_min: Tensor
    t_max: Tensor


class PMF(nn.Module):
    """Proposed CFM replacement; existing CFM/train/infer entry points stay unchanged."""

    def __init__(self, transformer: nn.Module, options: PMFOptions | None = None,
                 vocab_char_map: dict[str, int] | None = None,
                 aux_mel_loss: nn.Module | None = None):
        super().__init__()
        self.transformer = transformer
        self.options = options or PMFOptions()
        self.vocab_char_map = vocab_char_map
        self.aux_mel_loss = aux_mel_loss
        self.target_sample_rate = self.options.sample_rate
        self.wav_frame_len = self.options.wav_frame_len

    # ---------------- Core functions: training ----------------

    def forward(self, inp: Tensor, text: Tensor | list[str], *,
                lens: Tensor | None = None, noise_scheduler=None):
        """[B,N] unscaled wave + text + lengths -> loss, cond, u, metrics.

        Keep Trainer's four-result contract. Metrics include flow_loss,
        loss_u, loss_v, aux_mel_loss, total_loss (all scalar).
        """
        if noise_scheduler is not None:
            raise ValueError('PMF uses its own paired-time schedule')
        x, condition = self.prepare_training_batch(inp, text, lens)
        schedule = self.sample_schedule(x.shape[0], x.device)
        z, vt = self.make_noisy_pair(x, condition.valid_mask, schedule.t)
        vg, vc = self.build_guided_target(z, vt, condition, schedule)
        condition, vg = self.apply_condition_dropout(condition, vg, vt)
        u, du_dt, v = self.predict_with_jvp(z, condition, schedule, vc)
        V = u + (schedule.t - schedule.r)[:, None] * du_dt.detach()
        loss_u, loss_v = self.compute_flow_losses(V, v, vg.detach(), condition)
        x_pred = z - schedule.t[:, None] * u
        mel = self.compute_mel_loss(x_pred, x, condition, schedule.t)
        flow_loss = loss_u + loss_v
        total = flow_loss + mel
        return total, condition.audio, u, dict(
            total_loss=total, flow_loss=flow_loss, loss_u=loss_u,
            loss_v=loss_v, aux_mel_loss=mel,
        )

    def predict_velocity(self, z: Tensor, condition: WaveCondition, t: Tensor,
                         h: Tensor, omega: Tensor, t_min: Tensor, t_max: Tensor):
        """[B,N] state + fixed condition + [B] scalars -> u,v [B,N].

        PMFDiT returns CLEAN waveform predictions in scaled amplitude units.
        This is the only place that converts those predictions into velocities.
        """
        x_u, x_v = self.transformer(
            x=z, cond=condition.audio, text=condition.text, time=t, h=h,
            omega=omega, t_min=t_min, t_max=t_max,
            mask=condition.valid_mask, lens=condition.lengths,
            drop_mask=condition.drop_mask,
        )
        denom = t.clamp(self.options.t_eps, 1.0)[:, None]
        u, v = (z - x_u) / denom, (z - x_v) / denom
        return (u.masked_fill(~condition.valid_mask, 0),
                v.masked_fill(~condition.valid_mask, 0))

    def predict_with_jvp(self, z: Tensor, condition: WaveCondition,
                         schedule: FlowSchedule, vc: Tensor):
        """State/condition/schedule + detached vc [B,N] -> u,du_dt,v [B,N]."""
        def field(z, t, r):
            return self.predict_velocity(z, condition, t, t-r, schedule.omega,
                                         schedule.t_min, schedule.t_max)

        # No random draws, condition changes or cache mutation inside field.
        return torch.func.jvp(
            field, (z, schedule.t, schedule.r),
            (vc.detach(), torch.ones_like(schedule.t), torch.zeros_like(schedule.r)),
            has_aux=True,
        )

    # ---------------- Core functions: inference ----------------

    @torch.no_grad()
    def sample(self, cond: Tensor, text: Tensor | list[str], duration: int | Tensor,
               *, lens: Tensor | None = None, steps: int = 1, omega: float = 1.0,
               t_min: float = 0.0, t_max: float = 1.0, seed: int | None = None,
               **legacy_options):
        """Unscaled reference [B,Nref] + text + duration in samples -> wave, trajectory.

        wave [B,Nmax] is unscaled with reference restored and padding zeroed.
        trajectory [steps+1,B,Nmax] is in scaled model units.
        Old cfg_strength/EPSS/sway options need explicit caller migration.
        """
        if legacy_options:
            raise ValueError(f'Unsupported legacy CFM sampling options: {sorted(legacy_options)}')
        if steps < 1:
            raise ValueError('steps must be positive')
        self.eval()
        z, condition = self.prepare_sampling(cond, text, duration, lens, seed)
        times = torch.linspace(1, 0, steps+1, device=z.device, dtype=z.dtype)
        trajectory = [z]
        for t, r in zip(times[:-1], times[1:]):
            z = self.sample_step(z, condition, t, r, omega, t_min, t_max)
            trajectory.append(z)
        wave = self.finalize_waveform(z, condition)
        return wave, torch.stack(trajectory)

    def sample_step(self, z: Tensor, condition: WaveCondition, t: Tensor, r: Tensor,
                    omega: float, t_min: float, t_max: float):
        """Current [B,N] wave + scalar interval/guidance -> updated [B,N] wave."""
        def batch_scalar(value):
            return torch.as_tensor(value, device=z.device, dtype=z.dtype).expand(z.shape[0])

        t, r = batch_scalar(t), batch_scalar(r)
        u, _ = self.predict_velocity(z, condition, t, t-r, batch_scalar(omega),
                                     batch_scalar(t_min), batch_scalar(t_max))
        return z - (t-r)[:, None] * u

    # ---------------- Auxiliary functions: implementation pending ----------------

    def prepare_training_batch(self, inp, text, lens) -> tuple[Tensor, WaveCondition]:
        """Unscaled [B,N], text, lengths -> scaled clean x [B,N], fixed condition.

        Reuse CFM tokenization, valid lengths and aligned random-span masking.
        Padding is zero. prediction_mask is a subset of valid_mask.
        """
        raise NotImplementedError('Port CFM batch preparation without its flow/time logic')

    def sample_schedule(self, batch: int, device) -> FlowSchedule:
        """Batch/device -> paired times, FM mask, omega and guidance interval [B]."""
        raise NotImplementedError('Sample paired logit-normal times and pMF guidance')

    def make_noisy_pair(self, x, valid_mask, t) -> tuple[Tensor, Tensor]:
        """Scaled clean x [B,N], valid mask, t [B] -> z,vt [B,N].

        e=noise_scale*normal; z=(1-t)*x+t*e; vt=(z-x)/clamp(t).
        Only padding is zeroed; the known reference region is still part of z.
        """
        raise NotImplementedError('Build the noise-to-data path with noise at t=1')

    @torch.no_grad()
    def build_guided_target(self, z, vt, condition, schedule) -> tuple[Tensor, Tensor]:
        """z,vt,condition,schedule -> detached vg,vc [B,N].

        TODO call predict_velocity with h=0 and interval [0,1] for conditional
        and null audio/text. Null branch uses omega=1. Recompute vc with the
        interval-adjusted omega; FM samples use full-interval guidance targets.
        """
        raise NotImplementedError('Construct pMF guidance for joint audio/text conditions')

    def apply_condition_dropout(self, condition, vg, vt) -> tuple[WaveCondition, Tensor]:
        """Condition, vg,vt -> condition with fixed drop_mask [B], adjusted vg.

        Jointly drop text/audio; dropped target is vt. Preserve vc from before
        dropout, as in pMF. Draw the mask once, outside the JVP closure.
        """
        raise NotImplementedError('Sample prefix-drop count and freeze the condition mask')

    def compute_flow_losses(self, V, v, vg, condition) -> tuple[Tensor, Tensor]:
        """Three [B,N] velocities + masks -> two scalar adaptive velocity losses.

        Proposed baseline: sum squared error per example over prediction_mask,
        divide by detached (sum+eps)**p, then average valid examples. Empty
        prediction regions must be excluded. Report raw MSE separately later.
        """
        raise NotImplementedError('Implement masked per-example pMF objectives')

    def compute_mel_loss(self, x_pred, x, condition, t) -> Tensor:
        """Scaled predicted/clean [B,N], condition, t -> scalar Mel loss.

        First baseline: reuse WavTTS Mel loss on unscaled masked audio, keeping
        its current weighting. No image LPIPS/ConvNeXt or extra adaptive Mel
        weighting. Return zero if aux_mel_loss is None.
        """
        raise NotImplementedError('Connect existing WavTTS masked Mel loss')

    def prepare_sampling(self, cond, text, duration, lens, seed) -> tuple[Tensor, WaveCondition]:
        """Reference/text/durations -> scaled initial noise [B,Nmax], condition.

        Reuse CFM reference preparation and length conventions. drop_mask=False;
        prediction_mask marks valid non-reference samples. Keep noise in all
        valid positions and zeros in padding; no reference clamping per step.
        """
        raise NotImplementedError('Prepare waveform infilling and initial noise')

    def finalize_waveform(self, z, condition) -> Tensor:
        """Final scaled z [B,Nmax] + condition -> unscaled, restored, padded wave.

        Restore known reference samples only here, divide by latents_scale,
        and zero padding beyond each valid length. Preserve the batch shape.
        """
        raise NotImplementedError('Restore reference audio and output waveform units')
