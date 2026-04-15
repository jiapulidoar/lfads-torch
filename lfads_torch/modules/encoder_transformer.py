import math
import torch
import torch.nn.functional as F
from torch import nn

from .initializers import init_linear_
from .recurrent import BidirectionalClippedGRU
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with optional time offset.

    The offset allows a sub-sequence (e.g. the CI segment) to carry
    absolute position information from the original full trial, rather
    than restarting at position 0 after the IC/CI split.
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x      : (batch, seq_len, d_model)
            offset : starting position index (use hps.ic_enc_seq_len for CI)
        """
        x = x + self.pe[:, offset : offset + x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────────────
# Single-direction Transformer encoder
# ──────────────────────────────────────────────────────────────────────────────

class _UnidirectionalTransformerEncoder(nn.Module):
    """
    Internal building block.  Maps (B, T, input_size) → (B, T, hidden_size).

    Also exposes a CLS token output for sequence summarisation (used by the
    IC encoder path if you ever want to switch IC to Transformer too).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        assert hidden_size % nhead == 0, (
            f"hidden_size ({hidden_size}) must be divisible by nhead ({nhead})"
        )
        self.input_proj = nn.Linear(input_size, hidden_size)

        # CLS token — useful for IC summarisation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(hidden_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_offset: int = 0,
        attn_mask: torch.Tensor = None,
    ):
        """
        Args:
            x          : (B, T, input_size)
            pos_offset : absolute time offset for positional encoding
            attn_mask  : optional (T+1, T+1) attention mask (True = ignore)
        Returns:
            cls_out : (B, hidden_size)
            seq_out : (B, T, hidden_size)
        """
        B = x.size(0)
        x = self.input_proj(x)                              # (B, T, H)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, H)
        x = torch.cat([cls, x], dim=1)                      # (B, T+1, H)
        # Offset by 1 because position 0 is occupied by the CLS token
        x = self.pos_enc(x, offset=pos_offset)
        x = self.transformer(x, mask=attn_mask)               # (B, T+1, H)
        return x[:, 0, :], x[:, 1:, :]                     # cls, seq


# ──────────────────────────────────────────────────────────────────────────────
# Bidirectional Transformer encoder (CI encoder)
# ──────────────────────────────────────────────────────────────────────────────

class BidirectionalTransformerEncoder(nn.Module):
    """
    Runs two independent Transformer encoders:
      - fwd_enc : on the sequence in natural order
      - bwd_enc : on the time-reversed sequence (then flipped back)

    This gives truly directional streams so that the lag-padding logic
    inherited from the original LFADS CI encoder is meaningful.

    Output shapes mirror the bidirectional GRU convention:
        fwd : (B, T, hidden_size)
        bwd : (B, T, hidden_size)
    which are concatenated to (B, T, hidden_size * 2) by the caller.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,       # per-direction hidden size = ci_enc_dim
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fwd_enc = _UnidirectionalTransformerEncoder(
            input_size, hidden_size, nhead, num_layers, dim_feedforward, dropout
        )
        self.bwd_enc = _UnidirectionalTransformerEncoder(
            input_size, hidden_size, nhead, num_layers, dim_feedforward, dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_offset: int = 0,
    ):
        """
        Args:
            x          : (B, T, input_size)
            pos_offset : absolute time offset (pass hps.ic_enc_seq_len)
        Returns:
            fwd_seq : (B, T, hidden_size)   causal stream
            bwd_seq : (B, T, hidden_size)   anti-causal stream
        """
        # Forward stream — natural order
        _, fwd_seq = self.fwd_enc(x, pos_offset=pos_offset)

        # Backward stream — reverse time, encode, flip back
        x_rev = torch.flip(x, dims=[1])
        _, bwd_seq = self.bwd_enc(x_rev, pos_offset=pos_offset)
        bwd_seq = torch.flip(bwd_seq, dims=[1])

        return fwd_seq, bwd_seq


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid LFADS Encoder
# ──────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Drop-in replacement for the original LFADS Encoder.

    IC path  — BidirectionalClippedGRU (unchanged)
        Short segment, single-vector output; GRU is the right tool here.

    CI path  — BidirectionalTransformerEncoder (new)
        Full-length segment, step-aligned output; Transformer captures
        long-range dependencies and provides truly bidirectional context
        at every timestep without the hidden-state approximation of a GRU.

    Output signature is identical to the original Encoder:
        ic_mean : (B, ic_dim)
        ic_std  : (B, ic_dim)
        ci      : (B, recon_seq_len, ci_enc_dim * 2)  or  (B, T, 0)
    """

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hps = hparams

        # ── IC encoder: GRU (original, untouched) ─────────────────────────
        self.ic_enc_h0 = nn.Parameter(
            torch.zeros(2, 1, hps.ic_enc_dim, requires_grad=True)
        )
        self.ic_enc = BidirectionalClippedGRU(
            input_size=hps.encod_data_dim,
            hidden_size=hps.ic_enc_dim,
            clip_value=hps.cell_clip,
        )
        self.ic_linear = nn.Linear(hps.ic_enc_dim * 2, hps.ic_dim * 2)
        init_linear_(self.ic_linear)

        # ── CI encoder: Transformer (new) ─────────────────────────────────
        self.use_con = all([
            hps.ci_enc_dim > 0,
            hps.con_dim > 0,
            hps.co_dim > 0,
        ])
        if self.use_con:
            self.ci_enc = BidirectionalTransformerEncoder(
                input_size=hps.encod_data_dim,
                hidden_size=hps.ci_enc_dim,         # per-direction
                nhead=getattr(hps, "ci_enc_nhead", 4),
                num_layers=getattr(hps, "ci_enc_layers", 2),
                dim_feedforward=getattr(
                    hps, "ci_enc_ffn_dim", hps.ci_enc_dim * 4
                ),
                dropout=hps.dropout_rate,
            )

        self.dropout = nn.Dropout(hps.dropout_rate)

    # ------------------------------------------------------------------ #
    def forward(self, data: torch.Tensor):
        hps = self.hparams
        batch_size = data.shape[0]

        assert data.shape[1] == hps.encod_seq_len, (
            f"Sequence length in HPs ({hps.encod_seq_len}) must match "
            f"data dim 1 ({data.shape[1]})."
        )

        data_drop = self.dropout(data)

        # Split into IC and CI segments
        if hps.ic_enc_seq_len > 0:
            ic_enc_data = data_drop[:, : hps.ic_enc_seq_len, :]
            ci_enc_data = data_drop[:, hps.ic_enc_seq_len :, :]
        else:
            ic_enc_data = data_drop
            ci_enc_data = data_drop

        # ── IC branch: GRU ────────────────────────────────────────────────
        ic_enc_h0 = self.ic_enc_h0.expand(-1, batch_size, -1).contiguous()
        _, h_n = self.ic_enc(ic_enc_data, ic_enc_h0)
        h_n = torch.cat([*h_n], dim=1)                      # (B, ic_enc_dim*2)
        h_n_drop = self.dropout(h_n)
        ic_params = self.ic_linear(h_n_drop)                # (B, ic_dim*2)
        ic_mean, ic_logvar = torch.split(ic_params, hps.ic_dim, dim=1)
        ic_std = torch.sqrt(torch.exp(ic_logvar) + hps.ic_post_var_min)

        # ── CI branch: Transformer ────────────────────────────────────────
        if self.use_con:
            # Pass the absolute time offset so positional encoding is correct
            ci_fwd, ci_bwd = self.ci_enc(
                ci_enc_data,
                pos_offset=hps.ic_enc_seq_len,
            )                           # each (B, T_ci, ci_enc_dim)

            # Lag: prevent controller from seeing the present too directly.
            # Identical logic to the original encoder — now meaningful because
            # fwd/bwd are genuinely causal/anti-causal streams.
            ci_fwd = F.pad(ci_fwd, (0, 0, hps.ci_lag, 0, 0, 0))
            ci_bwd = F.pad(ci_bwd, (0, 0, 0, hps.ci_lag, 0, 0))

            ci_len = hps.encod_seq_len - hps.ic_enc_seq_len
            ci = torch.cat(
                [ci_fwd[:, :ci_len, :], ci_bwd[:, -ci_len:, :]], dim=2
            )                           # (B, ci_len, ci_enc_dim*2)

            # Pad for forward prediction beyond the encoded window
            fwd_steps = hps.recon_seq_len - hps.encod_seq_len
            ci = F.pad(ci, (0, 0, 0, fwd_steps, 0, 0))

            # Restore IC-segment prefix with zeros (encoder didn't see it)
            ci = F.pad(ci, (0, 0, hps.ic_enc_seq_len, 0, 0, 0))
            #   → final shape: (B, recon_seq_len, ci_enc_dim*2)
        else:
            ci = torch.zeros(batch_size, hps.recon_seq_len, 0).to(data.device)

        return ic_mean, ic_std, ci