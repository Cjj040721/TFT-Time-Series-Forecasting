"""
Author     : Cao Junjia

Informer Wrapper — adapts the third_party/informer2020 model to the unified
forward signature used across this project.

The wrapper keeps the original Informer code UNTOUCHED in third_party/ and
simply adds:
  1. sys.path manipulation so imports work without installation
  2. A standard forward(enc_x, enc_t, dec_x, dec_t) → (B, pred_len) interface
  3. Helper build_informer() that constructs a model from a config dict

Requirements: third_party/informer2020/ must contain model.py (the Informer
class) and all its dependencies (attn, embed, encoder, decoder, masking).
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn


def _add_informer_to_path(third_party_root: str = "third_party/informer2020") -> None:
    abs_path = os.path.abspath(third_party_root)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)


class InformerWrapper(nn.Module):
    """
    Thin wrapper around the original Informer / InformerStack model.

    Parameters
    ----------
    cfg : dict with keys matching Informer.__init__ args:
        enc_in, dec_in, c_out, seq_len, label_len, pred_len,
        factor, d_model, n_heads, e_layers, d_layers, d_ff,
        dropout, attn, embed, freq, activation, output_attention,
        distil, mix
    third_party_root : path to informer2020 directory
    """

    def __init__(self, cfg: dict, third_party_root: str = "third_party/informer2020"):
        super().__init__()
        _add_informer_to_path(third_party_root)
        from models.model import Informer  # noqa: E402 — deferred import

        args = SimpleNamespace(**cfg)
        self.inner = Informer(
            enc_in           = cfg.get("enc_in", cfg.get("n_features", 7)),
            dec_in           = cfg.get("dec_in", cfg.get("n_features", 7)),
            c_out            = cfg.get("c_out", 1),
            seq_len          = cfg.get("seq_len", 96),
            label_len        = cfg.get("label_len", 48),
            out_len          = cfg.get("pred_len", 24),      # original Informer uses out_len, not pred_len
            factor           = cfg.get("factor", 5),
            d_model          = cfg.get("d_model", 512),
            n_heads          = cfg.get("n_heads", 8),
            e_layers         = cfg.get("e_layers", 3),
            d_layers         = cfg.get("d_layers", 2),
            d_ff             = cfg.get("d_ff", 2048),
            dropout          = cfg.get("dropout", 0.05),
            attn             = cfg.get("attn", "prob"),
            embed            = cfg.get("embed", "timeF"),
            freq             = cfg.get("freq", "h"),
            activation       = cfg.get("activation", "gelu"),
            output_attention = cfg.get("output_attention", False),
            distil           = cfg.get("distil", True),
            mix              = cfg.get("mix", True),
        )

        self.pred_len  = args.pred_len
        self.label_len = args.label_len

    def forward(
        self,
        enc_x: torch.Tensor,   # (B, seq_len, D)
        enc_t: torch.Tensor,   # (B, seq_len, F)
        dec_x: torch.Tensor,   # (B, label_len+pred_len, D)
        dec_t: torch.Tensor,   # (B, label_len+pred_len, F)
    ) -> torch.Tensor:
        """
        Returns
        -------
        pred : (B, pred_len)  — predicted target values (single variable)
        """
        # Zero out the future portion of dec_x to prevent leakage
        dec_x_masked = dec_x.clone()
        dec_x_masked[:, self.label_len:, :] = 0.0

        out = self.inner(enc_x, enc_t, dec_x_masked, dec_t)
        # out: (B, pred_len, c_out)  →  squeeze last dim if c_out==1
        if out.dim() == 3 and out.shape[-1] == 1:
            out = out.squeeze(-1)                       # (B, pred_len)
        elif out.dim() == 3:
            out = out[:, :, -1]                         # take target column
        return out


# ─── convenience constructor ──────────────────────────────────────────────────

def build_informer(
    n_features: int,
    seq_len:    int,
    label_len:  int,
    pred_len:   int,
    freq:       str = "h",
    c_out:      int = 1,
    third_party_root: str = "third_party/informer2020",
    **kwargs,
) -> InformerWrapper:
    cfg = dict(
        enc_in    = n_features,
        dec_in    = n_features,
        c_out     = c_out,
        seq_len   = seq_len,
        label_len = label_len,
        pred_len  = pred_len,
        freq      = freq,
        **kwargs,
    )
    return InformerWrapper(cfg, third_party_root=third_party_root)
