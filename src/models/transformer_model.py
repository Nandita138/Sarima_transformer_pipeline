import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, labels):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.labels[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, activation="gelu"):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


class TransformerAnomalyDetector:
    def __init__(self, config_path="config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_history = {"train_loss": [], "val_loss": []}
        self.optimal_threshold = self.config.get("anomaly_threshold", 0.5)

    @staticmethod
    def _load_config(config_path: Union[str, Dict]):
        if isinstance(config_path, dict):
            return config_path.get("transformer", {})
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("transformer", {})
        return {}

    def build_model(self, input_dim):
        self.model = TransformerEncoder(
            input_dim=input_dim,
            d_model=self.config.get("d_model", 64),
            nhead=self.config.get("nhead", 4),
            num_layers=self.config.get("num_encoder_layers", 3),
            dim_feedforward=self.config.get("dim_feedforward", 256),
            dropout=self.config.get("dropout", 0.1),
            activation=self.config.get("activation", "gelu"),
        ).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Transformer] Model built. Parameters: {total_params:,}")

    def train(self, train_data, val_data):
        X_train, _, labels_train = train_data
        X_val, _, labels_val = val_data
        input_dim = X_train.shape[2]
        self.build_model(input_dim)

        batch_size = self.config.get("batch_size", 64)
        train_loader = DataLoader(TimeSeriesDataset(X_train, labels_train, labels_train),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val, labels_val, labels_val),
                                batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.config.get("learning_rate", 0.001), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)

        epochs = self.config.get("epochs", 30)
        patience = self.config.get("patience", 7)
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"[Transformer] Training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, _, label_batch in train_loader:
                X_batch = X_batch.to(self.device)
                label_batch = label_batch.to(self.device).unsqueeze(1)
                optimizer.zero_grad()
                scores = self.model(X_batch)
                loss = criterion(scores, label_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            val_loss = self._evaluate_loss(val_loader, criterion)
            scheduler.step(val_loss)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("models_saved/transformer_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Transformer] Early stopping at epoch {epoch+1}")
                    break

        self._load_checkpoint("models_saved/transformer_best.pt")
        
        # Calibrate optimal anomaly threshold on validation split
        val_scores = self.predict(X_val)
        if len(np.unique(labels_val)) > 1:
            precisions, recalls, thresholds = precision_recall_curve(labels_val, val_scores)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            if best_idx < len(thresholds):
                self.optimal_threshold = float(thresholds[best_idx])
                print(f"[Transformer] Calibrated optimal anomaly threshold: {self.optimal_threshold:.4f} (Val F1: {f1_scores[best_idx]:.4f})")

        print(f"[Transformer] Training complete. Best val loss: {best_val_loss:.6f}")
        return self.training_history

    def _evaluate_loss(self, loader, criterion):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for X_batch, _, label_batch in loader:
                X_batch = X_batch.to(self.device)
                label_batch = label_batch.to(self.device).unsqueeze(1)
                scores = self.model(X_batch)
                total += criterion(scores, label_batch).item()
        return total / max(len(loader), 1)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained.")
        self.model.eval()
        loader = DataLoader(TimeSeriesDataset(X, np.zeros(len(X)), np.zeros(len(X))),
                            batch_size=128, shuffle=False)
        all_scores = []
        with torch.no_grad():
            for X_batch, _, _ in loader:
                X_batch = X_batch.to(self.device)
                scores = self.model(X_batch)
                all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores, axis=0).flatten()

    def predict_with_threshold(self, X, threshold=None):
        threshold = threshold if threshold is not None else self.optimal_threshold
        scores = self.predict(X)
        labels = (scores >= threshold).astype(int)
        return scores, labels

    def _save_checkpoint(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def _load_checkpoint(self, filepath):
        if Path(filepath).exists():
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))

    def save(self, filepath="models_saved/transformer_model.pt"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(), "config": self.config, "optimal_threshold": self.optimal_threshold}, filepath)
        print(f"[Transformer] Model saved to {filepath}")

    def load(self, filepath, input_dim):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.config = checkpoint["config"]
        self.optimal_threshold = checkpoint.get("optimal_threshold", 0.5)
        self.build_model(input_dim)
        self.model.load_state_dict(checkpoint["model_state"])
