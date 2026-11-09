"""Proposed dual-head waveform DiT interface; not yet a runnable backbone.

Keep WavTTS's waveform/text frontend and one-dimensional positions. Add h and
CFG conditioning to its modulation vector; return CLEAN wave predictions.
PMF.predict_velocity performs the velocity conversion, not this module.
"""
from __future__ import annotations

from torch import Tensor, nn


class PMFDiT(nn.Module):
    def __init__(self, backbone: nn.Module, *, aux_head_depth: int):
        """Existing WavTTS DiT + branch depth -> scaffold holding the base model.

        The current constructor only records the design inputs. TODO split
        base blocks into shared/u branches, create an independent v branch,
        add h/omega/interval embeddings and two output projections. Decide
        initialization explicitly; never alias u/v parameters or silently
        load an old single-head checkpoint as a full PMF checkpoint.
        """
        super().__init__()
        if not 0 < aux_head_depth <= backbone.depth:
            raise ValueError('aux_head_depth must be in [1, backbone.depth]')
        self.backbone = backbone
        self.dim = backbone.dim
        self.aux_head_depth = aux_head_depth

    # ---------------- Core function ----------------

    def forward(self, x: Tensor, cond: Tensor, text: Tensor, time: Tensor,
                h: Tensor, omega: Tensor, t_min: Tensor, t_max: Tensor, *,
                mask: Tensor, lens: Tensor, drop_mask: Tensor) -> tuple[Tensor, Tensor]:
        """x/cond [B,N], text [B,L], scalars [B], masks -> x_u,x_v [B,N].

        Both outputs are scaled clean-waveform predictions. Preserve input
        length after unpatchifying; padding must not affect valid tokens.
        This path must be deterministic and support torch.func.jvp.
        """
        tokens, token_mask, rope = self.prepare_tokens(x, cond, text, mask, lens, drop_mask)
        modulation = self.embed_flow_conditions(time, h, omega, t_min, t_max)
        shared = self.run_shared_blocks(tokens, modulation, token_mask, rope)
        u_tokens, v_tokens = self.run_output_heads(shared, modulation, token_mask, rope)
        return self.unpatchify_heads(u_tokens, v_tokens, x.shape[1])

    # ---------------- Auxiliary functions: implementation pending ----------------

    def prepare_tokens(self, x, cond, text, mask, lens, drop_mask):
        """Wave/text/conditions -> [B,S,D] tokens, [B,S] mask, 1-D RoPE.

        Reuse DiT._wav_to_tokens and input/text embedders. S=ceil(N/frame_len).
        Extend existing batch-wide drop booleans to fixed per-example masks.
        No mutable text cache or random dropout inside the JVP forward.
        """
        raise NotImplementedError('Reuse WavTTS frontend and apply per-example conditioning')

    def embed_flow_conditions(self, time, h, omega, t_min, t_max):
        """Five [B] scalars -> [B,D] modulation vector.

        Retain WavTTS time conditioning, and add h, 1-1/omega, t_min, t_max.
        This adapts the pMF objective to WavTTS, rather than copying image
        pMF's prefix-token architecture (which omits explicit t conditioning).
        """
        raise NotImplementedError('Add scalar embeddings and combine the modulation vector')

    def run_shared_blocks(self, tokens, modulation, token_mask, rope):
        """Tokens [B,S,D], modulation [B,D], mask/RoPE -> shared [B,S,D].

        TODO use depth-aux_head_depth blocks and a verified JVP-compatible
        attention backend. Disable activation checkpointing until verified.
        """
        raise NotImplementedError('Run the shared WavTTS DiT trunk')

    def run_output_heads(self, shared, modulation, token_mask, rope):
        """Shared [B,S,D] + conditioning -> two patch predictions [B,S,frame_len].

        Each branch has aux_head_depth blocks, final norm and projection.
        Do not share the final parameters between u and v.
        """
        raise NotImplementedError('Run independent u/v branches and project waveform patches')

    def unpatchify_heads(self, u_tokens, v_tokens, num_samples):
        """Two [B,S,frame_len] predictions + N -> x_u,x_v [B,N]."""
        raise NotImplementedError('Reuse DiT._tokens_to_wav for both heads')
