"""
Neural network module for stock price prediction.

Features:
- Dynamic input layer: automatically sizes to match the number of input features
- Adjustable hidden layers (default: 200, 100, 30)
- 6 output nodes: predicted % price change for each time horizon
  (1 day, 4 days, 1 week, 2 weeks, 1 month, 3 months)
- Output range: -100% to +100% (uses tanh activation scaled to [-1, 1])
- Weight saving/loading to/from files
"""

import os
import logging
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

import config

logger = logging.getLogger(__name__)


class StockPredictorNet(nn.Module):
    """
    Feedforward neural network for stock price prediction.

    Dynamically constructs layers based on the number of input features.
    Hidden layers are configurable (default: [200, 100, 30]).
    Output layer has 6 nodes, one per prediction horizon, outputting
    values in [-1, 1] representing predicted % change / 100.
    """

    def __init__(self, input_size: int,
                 hidden_layers: list[int] = None,
                 num_outputs: int = 6):
        """
        Args:
            input_size: Number of input features (set dynamically)
            hidden_layers: List of hidden layer sizes
            num_outputs: Number of output nodes (default 6 for 6 horizons)
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = list(config.HIDDEN_LAYERS)

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layers
        self.num_outputs = num_outputs

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for i, h_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.2))
            prev_size = h_size

        # Output layer: 6 nodes, tanh to constrain to [-1, 1]
        layers.append(nn.Linear(prev_size, num_outputs))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"StockPredictorNet created: input={input_size}, "
            f"hidden={hidden_layers}, output={num_outputs}"
        )

    def _init_weights(self):
        """Xavier/Glorot initialization for linear layers."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tensor of shape (batch_size, 6) with values in [-1, 1]
        """
        return self.network(x)

    def get_architecture_info(self) -> dict:
        """Return a dict describing the network architecture."""
        return {
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layer_sizes,
            "num_outputs": self.num_outputs,
            "total_params": sum(
                p.numel() for p in self.parameters()
            ),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


class ModelManager:
    """
    Manages model creation, saving, and loading.

    Handles:
    - Creating a new model with the right input size
    - Saving weights + architecture metadata to disk
    - Loading weights from a previous run
    - Rebuilding the model if input size changes
    """

    def __init__(self, weights_dir: str = None):
        self.weights_dir = weights_dir or config.WEIGHTS_DIR
        os.makedirs(self.weights_dir, exist_ok=True)

    def _weights_path(self, ticker: str) -> str:
        """Get the weights file path for a ticker."""
        return os.path.join(self.weights_dir, f"{ticker}_weights.pth")

    def _meta_path(self, ticker: str) -> str:
        """Get the metadata file path for a ticker."""
        return os.path.join(self.weights_dir, f"{ticker}_meta.pth")

    def save_model(self, model: StockPredictorNet, ticker: str,
                   feature_names: list[str] = None,
                   training_info: dict = None):
        """
        Save model weights and architecture metadata.

        Args:
            model: The trained model
            ticker: Ticker symbol (used in filename)
            feature_names: List of feature names for reproducibility
            training_info: Optional dict of training stats
        """
        weights_path = self._weights_path(ticker)
        meta_path = self._meta_path(ticker)

        # Save model state dict
        torch.save(model.state_dict(), weights_path)

        # Save metadata
        meta = {
            "input_size": model.input_size,
            "hidden_layers": model.hidden_layer_sizes,
            "num_outputs": model.num_outputs,
            "feature_names": feature_names or [],
            "training_info": training_info or {},
        }
        torch.save(meta, meta_path)

        logger.info(f"Model saved for {ticker} -> {weights_path}")

    def load_model(self, ticker: str,
                   input_size: int = None,
                   hidden_layers: list[int] = None) -> Optional[StockPredictorNet]:
        """
        Load a previously saved model.

        If input_size is provided and differs from the saved model,
        a new model is created (weights are NOT loaded as they're
        incompatible).

        Args:
            ticker: Ticker symbol
            input_size: Expected input size (if known)
            hidden_layers: Hidden layer sizes (if overriding)

        Returns:
            Loaded StockPredictorNet, or None if no saved model exists
        """
        weights_path = self._weights_path(ticker)
        meta_path = self._meta_path(ticker)

        if not os.path.exists(weights_path):
            logger.info(f"No saved weights found for {ticker}")
            return None

        # Load metadata
        meta = {}
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, weights_only=False)

        saved_input_size = meta.get("input_size")
        saved_hidden = meta.get("hidden_layers", config.HIDDEN_LAYERS)

        # Check if architecture matches
        effective_input = input_size or saved_input_size
        effective_hidden = hidden_layers or saved_hidden

        if effective_input is None:
            logger.warning(
                f"Cannot load model for {ticker}: no input_size saved or provided"
            )
            return None

        if input_size and saved_input_size and input_size != saved_input_size:
            logger.warning(
                f"Input size mismatch for {ticker}: saved={saved_input_size}, "
                f"current={input_size}. Creating new model (weights discarded)."
            )
            return StockPredictorNet(input_size, effective_hidden)

        # Create model and load weights
        model = StockPredictorNet(effective_input, list(effective_hidden))
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()

        logger.info(
            f"Model loaded for {ticker}: input={effective_input}, "
            f"hidden={effective_hidden}"
        )
        return model

    def get_or_create_model(self, ticker: str, input_size: int,
                            hidden_layers: list[int] = None) -> StockPredictorNet:
        """
        Load existing model if compatible, or create a new one.

        Args:
            ticker: Ticker symbol
            input_size: Number of input features
            hidden_layers: Hidden layer config (optional override)

        Returns:
            StockPredictorNet ready for training or inference
        """
        model = self.load_model(ticker, input_size, hidden_layers)
        if model is not None:
            return model

        logger.info(f"Creating new model for {ticker} with input_size={input_size}")
        return StockPredictorNet(input_size, hidden_layers)

    def list_saved_models(self) -> list[str]:
        """List all tickers that have saved weights."""
        tickers = set()
        if os.path.exists(self.weights_dir):
            for f in os.listdir(self.weights_dir):
                if f.endswith("_weights.pth"):
                    tickers.add(f.replace("_weights.pth", ""))
        return sorted(tickers)
