"""
Author     : Cao Junjia

Interpretability utilities for TFT.

extract_attention_weights
    Runs the model on a batch, returns temporal attention (pred_len × seq_len).

extract_variable_importance
    Returns mean variable selection weights across the encoder.

Both functions work on CPU or GPU, and return numpy arrays for easy plotting.
"""
import numpy as np
import torch


def extract_attention_weights(
    model,
    enc_x:  torch.Tensor,
    enc_t:  torch.Tensor,
    dec_x:  torch.Tensor,
    dec_t:  torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Forward one batch through a TFT model and return the temporal
    attention matrix averaged over the batch.

    Parameters
    ----------
    model          : TemporalFusionTransformer instance
    enc_x, enc_t,
    dec_x, dec_t   : (B, T, D) tensors

    Returns
    -------
    attn_mean : (pred_len, seq_len) numpy array — averaged over batch & heads
    """
    model.eval()
    with torch.no_grad():
        enc_x = enc_x.to(device)
        enc_t = enc_t.to(device)
        dec_x = dec_x.to(device)
        dec_t = dec_t.to(device)
        _ = model(enc_x, enc_t, dec_x, dec_t)
        attn = model.last_attention          # (B, pred_len, seq_len)

    if attn is None:
        raise ValueError("Model did not produce attention weights. "
                         "Only TemporalFusionTransformer exposes last_attention.")

    return attn.mean(dim=0).cpu().numpy()    # (pred_len, seq_len)


def extract_variable_importance(
    model,
    enc_x:  torch.Tensor,
    enc_t:  torch.Tensor,
    dec_x:  torch.Tensor,
    dec_t:  torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Returns mean variable selection weights across time and batch.

    Returns
    -------
    var_importance : (n_vars,) numpy array, sums to ~1
    """
    model.eval()
    with torch.no_grad():
        enc_x = enc_x.to(device)
        enc_t = enc_t.to(device)
        dec_x = dec_x.to(device)
        dec_t = dec_t.to(device)
        _ = model(enc_x, enc_t, dec_x, dec_t)
        var_w = model.last_var_weights       # (B, seq_len, n_vars)

    if var_w is None:
        raise ValueError("Model did not produce variable weights.")

    return var_w.mean(dim=(0, 1)).cpu().numpy()   # (n_vars,)
