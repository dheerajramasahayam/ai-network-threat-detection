from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.unsqueeze(-1)
        _, (hidden, _) = self.lstm(sequence)
        return self.head(hidden[-1])


class LSTMThreatDetector:
    model_name = "LSTM"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        epochs: int = 8,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _LSTMClassifier(input_dim=input_dim, num_classes=num_classes).to(self.device)
        self.history: dict[str, list[float]] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, list[float]]:
        torch.manual_seed(self.random_state)
        class_counts = np.bincount(y_train, minlength=self.num_classes)
        class_weights = len(y_train) / (self.num_classes * np.clip(class_counts, 1, None))

        sample_weights = torch.tensor(class_weights[y_train], dtype=torch.double)
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        )

        best_state = deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        patience = 0
        history = {"loss": [], "val_loss": []}

        for _ in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_features.size(0)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()

            epoch_train_loss = train_loss / len(train_loader.dataset)
            history["loss"].append(epoch_train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    break

        self.model.load_state_dict(best_state)
        self.history = history
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probabilities = []
        self.model.eval()
        with torch.no_grad():
            for (batch_features,) in loader:
                batch_features = batch_features.to(self.device)
                logits = self.model(batch_features)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.append(probs)
        return np.vstack(probabilities)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "LSTMThreatDetector":
        payload = torch.load(path, map_location="cpu")
        instance = cls(
            input_dim=payload["input_dim"],
            num_classes=payload["num_classes"],
            epochs=payload["epochs"],
            batch_size=payload["batch_size"],
            learning_rate=payload["learning_rate"],
            random_state=payload["random_state"],
        )
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.to(instance.device)
        instance.model.eval()
        return instance
