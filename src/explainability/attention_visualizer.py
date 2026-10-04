"""
Cross-Temporal Attention Map Visualizer and Extractor.
Extracts attention weights from M-RA-BiLSTM (Engine 2) to interpret
how future calendar queries attend to historical seasonal patterns.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def extract_cross_attention_weights(
    model,
    dataloader,
    device: str = "cpu",
    max_batches: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts attention weights from MRABiLSTM model across evaluation batches.

    Parameters
    ----------
    model : MRABiLSTM
        Trained deep learning model.
    dataloader : DataLoader
        DataLoader yielding (x_past, x_future, x_static, y_arr, y_nig, regime_id).
    device : str
        Inference device ('cpu' or 'cuda').
    max_batches : int
        Maximum number of evaluation batches to process.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (attn_weights_mean, regime_ids_all)
        attn_weights_mean: shape (H, L) average attention matrix.
        regimes: shape (N,) regime labels for disaggregation.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to extract attention weights.")

    model.eval()
    model.to(device)

    all_attns = []
    all_regimes = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            x_past = batch[0].to(device)
            x_future = batch[1].to(device)
            x_static = batch[2].to(device)
            regime_id = batch[5].to(device)

            # Invoke forward with return_attn=True
            _, _, attn = model(x_past, x_future, x_static, regime_id, return_attn=True)
            all_attns.append(attn.cpu().numpy())
            all_regimes.append(regime_id.cpu().numpy())

    if len(all_attns) == 0:
        return np.zeros((12, 24)), np.zeros(0)

    concatenated_attns = np.concatenate(all_attns, axis=0)  # (N, H, L)
    concatenated_regimes = np.concatenate(all_regimes, axis=0)  # (N,)

    return concatenated_attns, concatenated_regimes


class AttentionVisualizer:
    """
    Generates academic attention heatmaps and quantitative attention concentration metrics.
    """
    def __init__(self, forecast_horizon: int = 12, lookback_window: int = 24):
        self.H = forecast_horizon
        self.L = lookback_window

    def compute_seasonal_concentration(self, attn_matrix: np.ndarray) -> Dict[str, float]:
        """
        Quantifies the proportion of attention concentrated on 12-month and 24-month seasonal lags.
        """
        # Historical indices: -24 to -1
        # lag -12 corresponds to column L - 12, lag -24 corresponds to column 0 (if L=24)
        total_mass = np.sum(attn_matrix) + 1e-12
        if self.L >= 12:
            mass_lag_12 = np.sum(attn_matrix[:, self.L - 12])
        else:
            mass_lag_12 = 0.0

        if self.L >= 24:
            mass_lag_24 = np.sum(attn_matrix[:, 0])
        else:
            mass_lag_24 = 0.0

        return {
            "seasonal_lag12_ratio": float(mass_lag_12 / total_mass),
            "seasonal_lag24_ratio": float(mass_lag_24 / total_mass),
            "combined_seasonal_ratio": float((mass_lag_12 + mass_lag_24) / total_mass),
        }

    def plot_attention_heatmap(
        self,
        attn_matrix: np.ndarray,
        save_path: str,
        title: str = "Cross-Temporal Attention Weights (Query: Future Months vs Key: Historical Lags)",
        cmap: str = "Blues",
    ) -> str:
        """
        Renders a publication-quality heatmap for cross-temporal attention.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        im = ax.imshow(attn_matrix, aspect="auto", cmap=cmap, interpolation="nearest")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention Weight $\\alpha_{h, t}$", fontsize=11)

        # Ticks and Labels
        ax.set_yticks(np.arange(self.H))
        ax.set_yticklabels([f"Horizon t+{h+1}" for h in range(self.H)], fontsize=9)

        ax.set_xticks(np.arange(0, self.L, 2))
        ax.set_xticklabels([f"Lag -{self.L - t}" for t in range(0, self.L, 2)], fontsize=9, rotation=45)

        ax.set_ylabel("Query: Future Calendar Steps ($H$)", fontsize=12)
        ax.set_xlabel("Key/Value: Historical Memory Horizon ($L$)", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return save_path
