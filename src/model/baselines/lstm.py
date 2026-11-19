"""
Author     : Cao Junjia

LSTM Baseline for multi-step time-series forecasting.

Architecture:
    Encoder: multi-layer LSTM over enc_x  (seq_len, D)
    Decoder: FC head applied to final hidden state → pred_len scalars

The model predicts a single target variable (e.g. OT) for pred_len steps.
"""
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Parameters
    ----------
    n_features   : number of input feature dimensions (D)
    hidden_size  : LSTM hidden units
    num_layers   : stacked LSTM layers
    pred_len     : number of future steps to predict
    dropout      : dropout rate (applied between LSTM layers)
    """

    def __init__(
        self,
        n_features:  int,
        hidden_size: int = 256,
        num_layers:  int = 2,
        pred_len:    int = 24,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len

        self.encoder = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Project final hidden state → prediction sequence
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, pred_len),
        )

    def forward(self, enc_x: torch.Tensor, *args) -> torch.Tensor:
        """
        Parameters
        ----------
        enc_x : (B, seq_len, D)

        Returns
        -------
        pred  : (B, pred_len)
        """
        _, (h, _) = self.encoder(enc_x)   # h: (num_layers, B, H)
        h_last = h[-1]                     # (B, H)  — top-layer hidden
        h_last = self.dropout(h_last)
        pred = self.projection(h_last)     # (B, pred_len)
        return pred
