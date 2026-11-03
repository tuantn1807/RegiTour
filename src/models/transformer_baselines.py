"""Direct multi-horizon Transformer baselines for the tourism panel.

The forecasters use only the two target series (arrivals and overnight stays),
so they provide univariate-per-target/multivariate-channel baselines without
regime, economic, or calendar features. Each country/accommodation series is
windowed independently. Training labels are restricted to the training split;
validation selects the checkpoint and test forecasting proceeds in non-overlap-
ping horizon blocks without consuming observed test targets.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "PatchTST, iTransformer, and TimesNet baselines require PyTorch. "
        "Install the project dependencies with `pip install -r requirements.txt`."
    ) from exc


class _PatchTST(nn.Module):
    """Channel-independent patch transformer with a direct horizon head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int = 2,
        patch_len: int = 6,
        stride: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B,L,C -> (B*C),L; each target channel is forecast independently.
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch * channels, self.seq_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = self.value_embedding(patches) + self.position_embedding
        encoded = self.norm(self.encoder(patches))
        forecasts = self.head(encoded.flatten(start_dim=1))
        return forecasts.reshape(batch, channels, -1).transpose(1, 2)


class _iTransformer(nn.Module):
    """Inverted transformer: each target variable is a token."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, pred_len)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B,L,C -> B,C,L, projecting each variable's history into a token.
        tokens = self.value_embedding(x.transpose(1, 2))
        tokens = self.norm(self.encoder(tokens))
        forecasts = self.projection(tokens)  # B,C,H
        return forecasts.transpose(1, 2)


class _Inception2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernels = (1, 3, 5)
        self.convs = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2)
            for k in kernels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([conv(x) for conv in self.convs], dim=0).mean(dim=0)


class _TimesBlock(nn.Module):
    """TimesNet-style 2D period reshaping and multi-kernel convolution."""

    def __init__(self, d_model: int, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            _Inception2D(d_model),
            nn.GELU(),
            _Inception2D(d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        spectrum = torch.fft.rfft(x, dim=1)
        frequency_strength = spectrum.abs().mean(dim=(0, 2))
        if frequency_strength.numel() <= 1:
            return self.norm(x)

        nonzero_frequency_strength = frequency_strength[1:]
        count = min(self.top_k, nonzero_frequency_strength.numel())
        top_frequency = torch.topk(nonzero_frequency_strength, k=count).indices + 1
        periods = torch.clamp(seq_len // top_frequency, min=1)

        sample_weights = spectrum.abs().mean(dim=2)[:, top_frequency]
        sample_weights = torch.softmax(sample_weights, dim=1)
        period_outputs = []

        for period in periods.tolist():
            padded_len = ((seq_len + period - 1) // period) * period
            padded = F.pad(x, (0, 0, 0, padded_len - seq_len))
            grid = padded.reshape(batch, padded_len // period, period, d_model)
            grid = grid.permute(0, 3, 1, 2).contiguous()
            transformed = self.conv(grid)
            transformed = transformed.permute(0, 2, 3, 1).reshape(batch, padded_len, d_model)
            period_outputs.append(transformed[:, :seq_len, :])

        stacked = torch.stack(period_outputs, dim=-1)
        mixed = (stacked * sample_weights[:, None, None, :]).sum(dim=-1)
        return self.norm(x + mixed)


class _TimesNet(nn.Module):
    """Compact TimesNet forecaster using FFT-selected temporal periods."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int = 2,
        d_model: int = 32,
        e_layers: int = 2,
        top_k: int = 5,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_channels, d_model)
        self.blocks = nn.ModuleList(_TimesBlock(d_model, top_k=top_k) for _ in range(e_layers))
        self.temporal_projection = nn.Linear(seq_len, pred_len)
        self.output_projection = nn.Linear(d_model, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.temporal_projection(hidden.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(hidden)


class _PanelTransformerForecaster:
    """Shared panel preprocessing, training, validation, and recursive inference."""

    MODEL_CLASS = None
    MODEL_NAME = "Transformer"

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 12,
        epochs: int = 30,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        patience: int = 6,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net: Optional[nn.Module] = None
        self.scalers: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self.keys = ["geo", "accommodation_type"]
        self.targets = ["arrivals_total", "nights_total"]
        self.best_validation_loss: Optional[float] = None

    @staticmethod
    def _entity_key(geo, accommodation_type) -> Tuple[str, str]:
        return str(geo), str(accommodation_type)

    @staticmethod
    def _log_targets(frame: pd.DataFrame) -> np.ndarray:
        values = frame[["arrivals_total", "nights_total"]].to_numpy(dtype=np.float32)
        return np.log1p(np.clip(values, 0.0, None))

    def _scaled(self, values: np.ndarray, key: Tuple[str, str]) -> np.ndarray:
        mean, scale = self.scalers[key]
        return (values - mean) / scale

    def _make_validation_tensors(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        inputs, targets, masks = [], [], []
        train_groups = {
            self._entity_key(key[0], key[1]): group.sort_values("date")
            for key, group in train_df.groupby(self.keys, sort=False)
        }

        for key_values, val_group in val_df.groupby(self.keys, sort=False):
            key = self._entity_key(key_values[0], key_values[1])
            history = train_groups.get(key)
            if history is None or len(history) < self.lookback:
                continue
            val_group = val_group.sort_values("date")
            n_valid = min(len(val_group), self.horizon)
            x = self._scaled(self._log_targets(history.tail(self.lookback)), key)
            y_valid = self._scaled(self._log_targets(val_group.iloc[:n_valid]), key)
            y = np.zeros((self.horizon, 2), dtype=np.float32)
            mask = np.zeros((self.horizon, 1), dtype=np.float32)
            y[:n_valid] = y_valid
            mask[:n_valid] = 1.0
            inputs.append(x)
            targets.append(y)
            masks.append(mask)

        if not inputs:
            return None
        return (
            torch.tensor(np.asarray(inputs), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(targets), dtype=torch.float32, device=self.device),
            torch.tensor(np.asarray(masks), dtype=torch.float32, device=self.device),
        )

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        required = set(self.keys + self.targets + ["date"])
        missing = sorted(required - set(train_df.columns))
        if missing:
            raise ValueError(f"{self.MODEL_NAME} training data is missing columns: {missing}")

        torch.manual_seed(self.random_state)
        self.best_validation_loss = None
        self.scalers.clear()
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        train_df = train_df.sort_values(self.keys + ["date"])
        train_windows_x: List[np.ndarray] = []
        train_windows_y: List[np.ndarray] = []

        for key_values, group in train_df.groupby(self.keys, sort=False):
            key = self._entity_key(key_values[0], key_values[1])
            group = group.sort_values("date")
            log_values = self._log_targets(group)
            mean = log_values.mean(axis=0).astype(np.float32)
            scale = (log_values.std(axis=0) + 1e-5).astype(np.float32)
            self.scalers[key] = (mean, scale)
            scaled_values = (log_values - mean) / scale

            last_start = len(scaled_values) - self.lookback - self.horizon
            for start in range(last_start + 1):
                split = start + self.lookback
                train_windows_x.append(scaled_values[start:split])
                train_windows_y.append(scaled_values[split:split + self.horizon])

        if not train_windows_x:
            raise ValueError(
                f"No {self.MODEL_NAME} training windows were created. "
                f"Each series needs at least {self.lookback + self.horizon} training rows."
            )

        x_train = torch.tensor(np.asarray(train_windows_x), dtype=torch.float32)
        y_train = torch.tensor(np.asarray(train_windows_y), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.net = self.MODEL_CLASS(self.lookback, self.horizon).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        val_tensors = self._make_validation_tensors(train_df, val_df) if val_df is not None and len(val_df) else None
        best_state = None
        epochs_without_improvement = 0

        for _epoch in range(self.epochs):
            self.net.train()
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.net(bx)
                loss = F.mse_loss(prediction, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()

            if val_tensors is None:
                continue
            self.net.eval()
            with torch.no_grad():
                vx, vy, vm = val_tensors
                val_prediction = self.net(vx)
                val_loss = float((((val_prediction - vy) ** 2) * vm).sum() / (vm.sum() * 2).clamp_min(1.0))

            if self.best_validation_loss is None or val_loss < self.best_validation_loss - 1e-6:
                self.best_validation_loss = val_loss
                best_state = copy.deepcopy(self.net.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        history_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.net is None:
            raise RuntimeError(f"Call fit() before predicting with {self.MODEL_NAME}.")

        histories = {
            self._entity_key(key[0], key[1]): group.sort_values("date")
            for key, group in history_df.groupby(self.keys, sort=False)
        }
        lookup: Dict[Tuple[str, str, pd.Timestamp], Tuple[float, float]] = {}
        self.net.eval()

        for key_values, test_group in test_df.groupby(self.keys, sort=False):
            key = self._entity_key(key_values[0], key_values[1])
            history = histories.get(key)
            if history is None or key not in self.scalers:
                raise ValueError(f"No training/history data available for entity {key}.")

            test_group = test_group.sort_values("date")
            scaled_history = self._scaled(self._log_targets(history), key).tolist()
            dates = pd.to_datetime(test_group["date"]).tolist()
            cursor = 0
            while cursor < len(test_group):
                context = np.asarray(scaled_history[-self.lookback:], dtype=np.float32)
                if len(context) < self.lookback:
                    context = np.pad(context, ((self.lookback - len(context), 0), (0, 0)), mode="edge")
                x = torch.tensor(context[None, :, :], dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    forecast_scaled = self.net(x)[0].cpu().numpy()

                block_len = min(self.horizon, len(test_group) - cursor)
                mean, scale = self.scalers[key]
                forecast_log = forecast_scaled[:block_len] * scale + mean
                forecast_raw = np.clip(np.expm1(forecast_log), 0.0, None)
                for offset in range(block_len):
                    lookup[(key[0], key[1], pd.Timestamp(dates[cursor + offset]))] = (
                        float(forecast_raw[offset, 0]),
                        float(forecast_raw[offset, 1]),
                    )
                scaled_history.extend(forecast_scaled[:block_len].tolist())
                cursor += block_len

        keys = [
            (
                str(row.geo),
                str(row.accommodation_type),
                pd.Timestamp(row.date),
            )
            for row in test_df[["geo", "accommodation_type", "date"]].itertuples(index=False)
        ]
        missing = [key for key in keys if key not in lookup]
        if missing:
            raise ValueError(f"{self.MODEL_NAME} produced no forecast for {len(missing)} test rows.")
        predictions = np.asarray([lookup[key] for key in keys], dtype=np.float32)
        return predictions[:, 0], predictions[:, 1]


class PatchTSTForecaster(_PanelTransformerForecaster):
    MODEL_CLASS = _PatchTST
    MODEL_NAME = "PatchTST"


class iTransformerForecaster(_PanelTransformerForecaster):
    MODEL_CLASS = _iTransformer
    MODEL_NAME = "iTransformer"


class TimesNetForecaster(_PanelTransformerForecaster):
    MODEL_CLASS = _TimesNet
    MODEL_NAME = "TimesNet"
