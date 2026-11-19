"""
Author     : Cao Junjia

Temporal Fusion Transformer (TFT) implementation.
Includes GLU, GRN (Gated Residual Network), VSN (Variable Selection Network),
and interpretable multi-head attention for multi-step time-series forecasting.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """
    Gated Residual Network
    """

    def __init__(self, d_model: int, d_context: int = None, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc_ctx = nn.Linear(d_context, d_model, bias=False) if d_context else None
        self.glu = GLU(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        residual = x
        h = self.fc1(x)

        if c is not None and self.fc_ctx is not None:
            if c.dim() == 2 and h.dim() == 3:
                c = c.unsqueeze(1).expand(-1, h.shape[1], -1)
            h = h + self.fc_ctx(c)

        h = F.elu(h)
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network.
    Input : list of per-variable embeddings (one tensor per variable)
    Output: weighted fused representation + per-variable selection weights
    """

    def __init__(self, n_vars: int, d_model: int, d_context: int = None, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model

        self.var_grns = nn.ModuleList([
            GRN(d_model, dropout=dropout) for _ in range(n_vars)
        ])
        self.agg_grn = GRN(n_vars * d_model, d_context=d_context, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.w_proj = nn.Linear(n_vars * d_model, n_vars)

    def forward(self, embeddings: list, context: torch.Tensor = None):
        """
        embeddings: list of tensors, each shape (B, T, d_model)
        """
        assert len(embeddings) == self.n_vars, f"expected {self.n_vars} vars, got {len(embeddings)}"

        var_enc = [self.var_grns[i](embeddings[i]) for i in range(self.n_vars)]
        stacked = torch.cat(var_enc, dim=-1)  # (B, T, n_vars * d_model)

        ctx = None
        if context is not None:
            if context.dim() == 2 and stacked.dim() == 3:
                ctx = context.unsqueeze(1).expand(-1, stacked.shape[1], -1)
            else:
                ctx = context

        flat = self.agg_grn(stacked, ctx)     # (B, T, n_vars * d_model)
        weights = self.softmax(self.w_proj(flat))  # (B, T, n_vars)

        var_stack = torch.stack(var_enc, dim=-2)   # (B, T, n_vars, d_model)
        combined = (var_stack * weights.unsqueeze(-1)).sum(-2)  # (B, T, d_model)

        return combined, weights


class InterpretableMultiheadAttention(nn.Module):
    """
    TFT-style interpretable multi-head attention.
    - Q, K projected independently per head
    - V uses shared weights across heads
    - Supports T_q != T_k (encoder-decoder cross-attention)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_head)   # shared V
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """
        q: (B, T_q, d_model)
        k: (B, T_k, d_model)
        v: (B, T_k, d_model)
        """
        B, T_q, _ = q.shape
        _, T_k, _ = k.shape

        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T_q, d_h)
        K = self.W_k(k).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T_k, d_h)
        V = self.W_v(v).unsqueeze(1).expand(-1, self.n_heads, -1, -1)            # (B, H, T_k, d_h)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)    # (B, H, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_mean = attn.mean(dim=1)  # (B, T_q, T_k)

        ctx = torch.matmul(attn, V)   # (B, H, T_q, d_h)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.W_o(ctx)

        return out, attn_mean


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer with corrected decoder.

    Key fixes applied:
    1. Decoder uses both dec_x + dec_t instead of dec_t only
    2. Decoder VSN structure aligned with encoder
    3. Encoder final hidden state used as decoder static enrichment context
    """

    def __init__(
        self,
        n_hist_vars: int,
        n_future_vars: int,
        d_model: int = 128,
        n_heads: int = 4,
        lstm_layers: int = 2,
        pred_len: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.pred_len = pred_len
        self.n_hist = n_hist_vars
        self.n_fut = n_future_vars

        # Per-variable linear projection into d_model space
        self.hist_proj = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_hist_vars)
        ])
        self.fut_proj = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_future_vars)
        ])

        # encoder / decoder variable selection
        self.enc_vsn = VariableSelectionNetwork(
        n_vars=n_hist_vars + n_future_vars,
        d_model=d_model,
        dropout=dropout,
    )
        self.dec_vsn = VariableSelectionNetwork(
        n_vars=n_hist_vars + n_future_vars,
        d_model=d_model,
        dropout=dropout,
    )


        # LSTM encoder / decoder
        self.enc_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dec_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # post-LSTM gating
        self.gate_enc = GLU(d_model)
        self.gate_dec = GLU(d_model)
        self.norm_enc = nn.LayerNorm(d_model)
        self.norm_dec = nn.LayerNorm(d_model)

        # static enrichment
        self.static_enrich = GRN(d_model, d_context=d_model, dropout=dropout)

        # attention
        self.attn = InterpretableMultiheadAttention(d_model, n_heads, dropout)
        self.attn_gate = GLU(d_model)
        self.attn_norm = nn.LayerNorm(d_model)

        # output
        self.output_grn = GRN(d_model, dropout=dropout)
        self.output_proj = nn.Linear(d_model, 1)

        # for interpretability
        self._last_attn = None
        self._last_var_weights = None

    def _project_vars(self, x: torch.Tensor, proj_list: nn.ModuleList):
        """
        x: (B, T, n_vars)
        return: list of (B, T, d_model)
        """
        return [proj_list[i](x[..., i:i+1]) for i in range(len(proj_list))]

    def forward(
        self,
        enc_x: torch.Tensor,   # (B, seq_len, n_hist_vars)
        enc_t: torch.Tensor,   # (B, seq_len, n_future_vars)
        dec_x: torch.Tensor,   # (B, label_len + pred_len, n_hist_vars)
        dec_t: torch.Tensor,   # (B, label_len + pred_len, n_future_vars)
    ) -> torch.Tensor:
        # =========================
        # Encoder
        # =========================
        hist_emb_enc = self._project_vars(enc_x, self.hist_proj)
        fut_emb_enc = self._project_vars(enc_t, self.fut_proj)

        enc_selected, var_w = self.enc_vsn(hist_emb_enc + fut_emb_enc)
        self._last_var_weights = var_w

        enc_lstm_out, (h_enc, c_enc) = self.enc_lstm(enc_selected)
        enc_out = self.norm_enc(self.gate_enc(enc_lstm_out) + enc_selected)

        # static context from encoder
        static_ctx = h_enc[-1]  # (B, d_model)

        # Decoder: use both historical features (dec_x) and time features (dec_t)
        # Original bug: decoder used dec_t only, ignoring dec_x
        # Fix: concatenate both inputs through the VSN, matching encoder structure
        hist_emb_dec = self._project_vars(dec_x, self.hist_proj)
        fut_emb_dec = self._project_vars(dec_t, self.fut_proj)

        dec_selected, _ = self.dec_vsn(hist_emb_dec + fut_emb_dec)

        dec_lstm_out, _ = self.dec_lstm(dec_selected, (h_enc, c_enc))
        dec_out = self.norm_dec(self.gate_dec(dec_lstm_out) + dec_selected)

        # static enrichment
        dec_enriched = self.static_enrich(dec_out, static_ctx)

        # =========================
        # Cross attention
        # query = decoder future horizon
        # key/value = encoder outputs
        # =========================
        query = dec_enriched[:, -self.pred_len:, :]   # (B, pred_len, d_model)

        attn_out, attn_w = self.attn(query, enc_out, enc_out)
        self._last_attn = attn_w

        attn_out = self.attn_norm(self.attn_gate(attn_out) + query)

        # =========================
        # Output projection
        # =========================
        out = self.output_grn(attn_out)
        out = self.output_proj(out).squeeze(-1)   # (B, pred_len)

        return out

    @property
    def last_attention(self):
        return self._last_attn

    @property
    def last_var_weights(self):
        return self._last_var_weights
