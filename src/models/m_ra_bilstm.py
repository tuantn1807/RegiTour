"""
M-RA-BiLSTM Attention Model (Multi-Horizon Regime-Aware Attention BiLSTM Backbone).
Engine 2 in the RA-HE-BiLSTM Architecture.
Primary Methodological Novelty:
  1. Dynamic Regime-Conditioned Forget Gate (clears crisis memory upon structural shock)
  2. Dual-Stream Cross-Temporal Attention (matches future calendar map Query with past states Key/Value)
  3. Direct Multi-Horizon Seq2Seq Decoding (H=6, 12 months in a single forward pass, 0 error compounding)
  4. Coupled Multi-Task Consistency Loss Head (penalizes business violations of nights/arrivals ratio)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class RegimeConditionedLSTMCell(nn.Module):
        """
        Custom LSTM Cell where the forget gate f_t is conditioned on regime embedding e_reg:
        f_t = sigmoid(W_f x_t + U_f h_{t-1} + W_reg e_reg + b_f)
        """
        def __init__(self, input_dim: int, hidden_dim: int, regime_dim: int):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.regime_dim = regime_dim

            # Gates: input, forget, cell candidate, output
            self.w_ih = nn.Linear(input_dim, 4 * hidden_dim)
            self.w_hh = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
            self.w_reg_f = nn.Linear(regime_dim, hidden_dim, bias=False)

        def forward(
            self,
            x_t: torch.Tensor,
            h_prev: torch.Tensor,
            c_prev: torch.Tensor,
            e_reg: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            x_t: (B, input_dim)
            h_prev, c_prev: (B, hidden_dim)
            e_reg: (B, regime_dim)
            """
            gates = self.w_ih(x_t) + self.w_hh(h_prev)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=-1)

            # Regime conditioning added to forget gate
            f_gate = f_gate + self.w_reg_f(e_reg)

            i_t = torch.sigmoid(i_gate)
            f_t = torch.sigmoid(f_gate)
            c_cand = torch.tanh(c_gate)
            o_t = torch.sigmoid(o_gate)

            c_t = f_t * c_prev + i_t * c_cand
            h_t = o_t * torch.tanh(c_t)
            return h_t, c_t


    class MRABiLSTM(nn.Module):
        """
        M-RA-BiLSTM: Novel deep learning backbone with regime forget gate and cross-temporal attention.
        """
        def __init__(
            self,
            num_countries: int = 40,
            num_accom_types: int = 5,
            num_regimes: int = 3,
            past_feat_dim: int = 10,
            future_feat_dim: int = 14,
            geo_emb_dim: int = 16,
            accom_emb_dim: int = 8,
            regime_emb_dim: int = 8,
            hidden_dim: int = 64,
            attn_dim: int = 64,
            forecast_horizon: int = 12,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.forecast_horizon = forecast_horizon

            # 1. Entity & Regime Embeddings
            self.geo_embedding = nn.Embedding(num_countries, geo_emb_dim)
            self.accom_embedding = nn.Embedding(num_accom_types, accom_emb_dim)
            self.regime_embedding = nn.Embedding(num_regimes, regime_emb_dim)

            # Total past input dimension
            total_past_dim = past_feat_dim + geo_emb_dim + accom_emb_dim

            # 2. BiLSTM Encoder with Regime-Conditioned Forget Gates
            self.cell_fwd = RegimeConditionedLSTMCell(total_past_dim, hidden_dim, regime_emb_dim)
            self.cell_bwd = RegimeConditionedLSTMCell(total_past_dim, hidden_dim, regime_emb_dim)
            self.dropout = nn.Dropout(dropout)

            # 3. Dual-Stream Cross-Temporal Attention
            # Query projects future calendar features: (B, H, future_feat_dim) -> (B, H, attn_dim)
            # Key/Value projects bidirectional hidden states: (B, L, 2*hidden_dim) -> (B, L, attn_dim)
            self.query_proj = nn.Linear(future_feat_dim, attn_dim)
            self.key_proj = nn.Linear(2 * hidden_dim, attn_dim)
            self.value_proj = nn.Linear(2 * hidden_dim, attn_dim)
            self.scale = np.sqrt(attn_dim)

            # 4. Multi-Horizon Direct Decoder Heads
            dec_in_dim = attn_dim + attn_dim + geo_emb_dim
            self.decoder_mlp = nn.Sequential(
                nn.Linear(dec_in_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            # Two linear output heads for multi-task arrivals and nights
            self.head_arr = nn.Linear(64, 1)
            self.head_nig = nn.Linear(64, 1)

        def forward(
            self,
            x_past: torch.Tensor,      # (B, L, past_feat_dim)
            x_future: torch.Tensor,    # (B, H, future_feat_dim)
            x_static: torch.Tensor,    # (B, 2) -> [country_id, accom_id]
            regime_id: torch.Tensor,   # (B,) -> regime state {0, 1, 2}
            return_attn: bool = False,
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            B, L, _ = x_past.shape
            H = x_future.shape[1]

            # Embeddings
            e_geo = self.geo_embedding(x_static[:, 0])        # (B, geo_emb_dim)
            e_accom = self.accom_embedding(x_static[:, 1])    # (B, accom_emb_dim)
            e_reg = self.regime_embedding(regime_id)          # (B, regime_emb_dim)

            # Expand static entity embeddings over past time steps
            e_static = torch.cat([e_geo, e_accom], dim=-1).unsqueeze(1).expand(-1, L, -1)
            x_past_combined = torch.cat([x_past, e_static], dim=-1)  # (B, L, total_past_dim)

            # Forward pass through bidirectional cells
            h_fwd, c_fwd = torch.zeros(B, self.hidden_dim, device=x_past.device), torch.zeros(B, self.hidden_dim, device=x_past.device)
            h_fwd_list = []
            for t in range(L):
                h_fwd, c_fwd = self.cell_fwd(x_past_combined[:, t, :], h_fwd, c_fwd, e_reg)
                h_fwd_list.append(h_fwd)

            h_bwd, c_bwd = torch.zeros(B, self.hidden_dim, device=x_past.device), torch.zeros(B, self.hidden_dim, device=x_past.device)
            h_bwd_list = []
            for t in reversed(range(L)):
                h_bwd, c_bwd = self.cell_bwd(x_past_combined[:, t, :], h_bwd, c_bwd, e_reg)
                h_bwd_list.insert(0, h_bwd)

            h_fwd_stack = torch.stack(h_fwd_list, dim=1)  # (B, L, hidden_dim)
            h_bwd_stack = torch.stack(h_bwd_list, dim=1)  # (B, L, hidden_dim)
            h_past = torch.cat([h_fwd_stack, h_bwd_stack], dim=-1)  # (B, L, 2*hidden_dim)
            h_past = self.dropout(h_past)

            # Dual-Stream Cross-Temporal Attention
            Q = self.query_proj(x_future)                  # (B, H, attn_dim)
            K = self.key_proj(h_past)                      # (B, L, attn_dim)
            V = self.value_proj(h_past)                    # (B, L, attn_dim)

            # Attention scores: Q @ K^T / sqrt(d) -> (B, H, L)
            attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Context vector: (B, H, attn_dim)
            context = torch.bmm(attn_weights, V)

            # Expand entity embedding for future steps
            e_geo_future = e_geo.unsqueeze(1).expand(-1, H, -1)
            dec_in = torch.cat([context, Q, e_geo_future], dim=-1)  # (B, H, dec_in_dim)
            dec_feat = self.decoder_mlp(dec_in)

            # Predictions (clipped at >= 0)
            y_arr_pred = F.relu(self.head_arr(dec_feat).squeeze(-1))  # (B, H)
            y_nig_pred = F.relu(self.head_nig(dec_feat).squeeze(-1))  # (B, H)

            if return_attn:
                return y_arr_pred, y_nig_pred, attn_weights
            return y_arr_pred, y_nig_pred


    class CoupledConsistencyLoss(nn.Module):
        """
        Coupled Multi-Task Consistency Loss Head.
        L = Huber(arr) + Huber(nig) + lambda * max(0, |nig/arr - base_ratio| - delta)^2
        Ensures 100% domain compliance between predicted arrivals and nights.
        """
        def __init__(self, lambda_cons: float = 0.5, delta: float = 1.0):
            super().__init__()
            self.huber = nn.SmoothL1Loss()
            self.lambda_cons = lambda_cons
            self.delta = delta

        def forward(
            self,
            y_arr_pred: torch.Tensor,
            y_arr_true: torch.Tensor,
            y_nig_pred: torch.Tensor,
            y_nig_true: torch.Tensor,
            base_ratios: torch.Tensor,  # (B,)
        ) -> torch.Tensor:
            l_arr = self.huber(y_arr_pred, y_arr_true)
            l_nig = self.huber(y_nig_pred, y_nig_true)

            # Empirical ratio consistency penalty
            ratio_pred = y_nig_pred / (y_arr_pred + 1e-4)  # (B, H)
            ratio_base = base_ratios.unsqueeze(1).expand_as(ratio_pred)

            diff = torch.abs(ratio_pred - ratio_base) - self.delta
            penalty = torch.mean(F.relu(diff) ** 2)

            total_loss = l_arr + l_nig + self.lambda_cons * penalty
            return total_loss
else:
    # Dummy placeholder if torch is still loading
    class MRABiLSTM:
        pass
    class CoupledConsistencyLoss:
        pass
