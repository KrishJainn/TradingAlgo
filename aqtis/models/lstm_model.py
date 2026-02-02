"""
AQTIS LSTM Predictor.

PyTorch-based LSTM for time-series return prediction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """
    LSTM model for sequential time-series prediction.

    Uses sequences of technical indicator features to predict future returns.
    """

    def __init__(
        self,
        input_size: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 60,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = None
        self._scaler = None
        self._feature_names: List[str] = []
        self._fitted = False
        self._device = "cpu"

    def _build_model(self, input_size: int):
        """Build the LSTM model."""
        try:
            import torch
            import torch.nn as nn

            class LSTMNet(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=0.2 if num_layers > 1 else 0,
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 32),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(32, 1),
                    )

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_hidden = lstm_out[:, -1, :]
                    return self.fc(last_hidden).squeeze(-1)

            self._model = LSTMNet(input_size, self.hidden_size, self.num_layers)
            self._model.to(self._device)
            return True

        except ImportError:
            logger.warning("PyTorch not available for LSTM model")
            return False

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from flat feature arrays."""
        sequences = []
        targets = []
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i: i + self.sequence_length])
            targets.append(y[i + self.sequence_length])
        return np.array(sequences), np.array(targets)

    def train(self, features: pd.DataFrame, targets: pd.Series) -> Dict:
        """
        Train the LSTM model.

        Args:
            features: DataFrame of indicator features (rows are time steps).
            targets: Series of forward returns.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            return {"error": "PyTorch not available", "method": "skipped"}

        # Prepare data
        valid = features.dropna().index.intersection(targets.dropna().index)
        X_raw = features.loc[valid].values
        y_raw = targets.loc[valid].values

        if len(X_raw) < self.sequence_length + 50:
            return {"error": "Insufficient data for LSTM", "samples": len(X_raw)}

        self._feature_names = list(features.columns)
        self.input_size = X_raw.shape[1]

        # Scale features
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_raw)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_raw)

        # Train/val split (time-based)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # Build model
        if not self._build_model(self.input_size):
            return {"error": "Could not build LSTM model"}

        # Convert to tensors
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        # Train
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        for epoch in range(self.epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                predictions = self._model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        # Validation
        self._model.eval()
        with torch.no_grad():
            val_pred = self._model(torch.FloatTensor(X_val).to(self._device)).cpu().numpy()
            val_mse = float(np.mean((val_pred - y_val) ** 2))
            val_direction_acc = float(np.mean((val_pred > 0) == (y_val > 0)))

        self._fitted = True

        logger.info(
            f"LSTM trained: {len(X_train)} sequences, "
            f"Val MSE={val_mse:.6f}, Val Dir Acc={val_direction_acc:.2%}"
        )

        return {
            "samples": len(X_train),
            "val_samples": len(X_val),
            "final_train_loss": train_losses[-1] if train_losses else 0,
            "val_mse": val_mse,
            "val_direction_accuracy": val_direction_acc,
            "epochs_completed": self.epochs,
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict returns from feature DataFrame."""
        if not self._fitted:
            return np.zeros(len(features))

        try:
            import torch
        except ImportError:
            return np.zeros(len(features))

        X = features[self._feature_names].values if self._feature_names else features.values
        X_scaled = self._scaler.transform(X)

        # Need at least sequence_length rows
        if len(X_scaled) < self.sequence_length:
            return np.zeros(len(features))

        # Create sequences
        predictions = np.zeros(len(X_scaled))
        self._model.eval()
        with torch.no_grad():
            for i in range(self.sequence_length, len(X_scaled)):
                seq = X_scaled[i - self.sequence_length: i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self._device)
                pred = self._model(seq_tensor).cpu().item()
                predictions[i] = pred

        return predictions

    def predict_single(self, features_sequence: np.ndarray) -> float:
        """Predict from a single sequence of features."""
        if not self._fitted:
            return 0.0

        try:
            import torch
        except ImportError:
            return 0.0

        if len(features_sequence) < self.sequence_length:
            return 0.0

        seq = features_sequence[-self.sequence_length:]
        if self._scaler:
            seq = self._scaler.transform(seq)

        self._model.eval()
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self._device)
            return float(self._model(seq_tensor).cpu().item())

    def save(self, path: str):
        """Save model to disk."""
        try:
            import torch
            state = {
                "model_state": self._model.state_dict() if self._model else None,
                "scaler": self._scaler,
                "feature_names": self._feature_names,
                "fitted": self._fitted,
                "config": {
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "sequence_length": self.sequence_length,
                },
            }
            torch.save(state, path)
        except ImportError:
            pass

    def load(self, path: str):
        """Load model from disk."""
        try:
            import torch
            state = torch.load(path, map_location=self._device)
            self._scaler = state["scaler"]
            self._feature_names = state["feature_names"]
            self._fitted = state["fitted"]
            config = state["config"]
            self.input_size = config["input_size"]
            self.hidden_size = config["hidden_size"]
            self.num_layers = config["num_layers"]
            self.sequence_length = config["sequence_length"]
            if state["model_state"] and self._build_model(self.input_size):
                self._model.load_state_dict(state["model_state"])
        except (ImportError, FileNotFoundError):
            pass
