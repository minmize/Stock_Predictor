"""
Training mode module.

Training process:
1. Downloads historical data via Massive flat files (S3)
2. Processes data in 3-month sliding windows
3. For each window:
   a. Computes features for the window
   b. Gets sentiment score (or uses neutral for historical)
   c. Runs forward pass through neural net
   d. Compares predictions vs actual future returns (the targets)
   e. Computes loss and backpropagates
4. Saves weights to disk at the end of each run
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config
from data_fetcher import MassiveDataFetcher, normalize_dataframe
from sentiment import SentimentEvaluator, get_training_sentiment
from features import (
    build_feature_matrix,
    compute_targets,
    get_feature_count,
    FEATURE_COLUMNS,
    compute_technical_features,
)
from neural_net import StockPredictorNet, ModelManager

logger = logging.getLogger(__name__)


class StockTrainer:
    """
    Handles the full training pipeline for a stock ticker.

    Coordinates data fetching, feature engineering, training loop,
    and model persistence.
    """

    def __init__(self, ticker: str, hidden_layers: list[int] = None,
                 use_sentiment: bool = True):
        """
        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")
            hidden_layers: Optional override for hidden layer sizes
            use_sentiment: Whether to fetch and use sentiment scores
        """
        self.ticker = ticker
        self.hidden_layers = hidden_layers or list(config.HIDDEN_LAYERS)
        self.use_sentiment = use_sentiment

        self.fetcher = MassiveDataFetcher()
        self.model_manager = ModelManager()
        self.sentiment_evaluator = SentimentEvaluator() if use_sentiment else None

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fetch_training_data(self, start_year: int, start_month: int,
                            end_year: int, end_month: int,
                            use_cache: bool = True) -> Optional[any]:
        """
        Fetch historical data for training via flat files.

        Tries to load from local cache first. If not available,
        downloads from S3 and caches locally.

        Args:
            start_year: Start year for data
            start_month: Start month
            end_year: End year
            end_month: End month
            use_cache: Whether to use cached data

        Returns:
            Normalized pandas DataFrame, or None on failure
        """
        import pandas as pd

        if use_cache:
            cached = self.fetcher.load_cached_csv(self.ticker)
            if cached is not None:
                logger.info(f"Using cached data for {self.ticker}")
                return normalize_dataframe(cached)

        logger.info(
            f"Downloading flat file data for {self.ticker} "
            f"({start_year}/{start_month} -> {end_year}/{end_month})"
        )
        raw_df = self.fetcher.download_ticker_flat_files(
            self.ticker, start_year, start_month, end_year, end_month
        )

        if raw_df.empty:
            logger.error(f"No data fetched for {self.ticker}")
            return None

        # Cache to disk
        self.fetcher.save_flat_data_to_csv(raw_df, self.ticker)

        return normalize_dataframe(raw_df)

    def fetch_training_data_rest(self, start_date: str,
                                 end_date: str) -> Optional[any]:
        """
        Alternative: fetch training data via REST API instead of flat files.
        Useful for smaller date ranges or when S3 access isn't configured.

        Args:
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"

        Returns:
            Normalized pandas DataFrame
        """
        df = self.fetcher.fetch_rest_aggregates(
            self.ticker, start_date, end_date
        )
        if df.empty:
            return None
        return normalize_dataframe(df)

    def prepare_training_samples(self, df, sentiment_score: float = 0.0):
        """
        Slide a window across the historical data and build
        (feature_vector, target) pairs.

        For each position where we have enough lookback data AND
        enough future data to compute targets, create one sample.

        Args:
            df: Normalized DataFrame with full history
            sentiment_score: Sentiment score to use for all samples
                           (or will be varied per window if we have
                           historical sentiment)

        Returns:
            Tuple of (features_array, targets_array) as numpy arrays,
            or (None, None) if insufficient data
        """
        window = config.TRAINING_WINDOW_DAYS
        max_horizon = max(config.PREDICTION_HORIZONS.values())

        # We need at least window + some buffer for technical indicators
        # (26 for EMA26, 20 for BB, 14 for RSI)
        min_start = 30  # Buffer for indicator warm-up

        # Compute technical features on the full dataset once
        feat_df = compute_technical_features(df)
        feat_df = feat_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

        if len(feat_df) < window + max_horizon:
            logger.warning(
                f"Insufficient data for training: {len(feat_df)} rows, "
                f"need at least {window + max_horizon}"
            )
            return None, None

        features_list = []
        targets_list = []

        # Slide window through the data
        for i in range(window, len(feat_df) - max_horizon):
            # Extract the lookback window
            window_data = feat_df.iloc[i - window:i][FEATURE_COLUMNS].values
            window_data = np.nan_to_num(
                window_data, nan=0.0, posinf=1.0, neginf=-1.0
            )

            # Flatten and append sentiment
            flat = window_data.flatten()
            flat = np.append(flat, sentiment_score)
            features_list.append(flat)

            # Compute targets from the original df (using feat_df indices)
            target = compute_targets(feat_df, i)
            if target is None:
                features_list.pop()
                continue
            targets_list.append(target)

        if not features_list:
            return None, None

        features_array = np.array(features_list, dtype=np.float32)
        targets_array = np.array(targets_list, dtype=np.float32)

        logger.info(
            f"Prepared {len(features_array)} training samples, "
            f"feature dim = {features_array.shape[1]}"
        )
        return features_array, targets_array

    def train(self, features: np.ndarray, targets: np.ndarray,
              epochs: int = None, learning_rate: float = None,
              batch_size: int = None) -> dict:
        """
        Run the training loop.

        Args:
            features: Feature matrix (num_samples, input_size)
            targets: Target matrix (num_samples, 6)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size

        Returns:
            Dict with training statistics
        """
        epochs = epochs or config.EPOCHS_PER_WINDOW
        learning_rate = learning_rate or config.LEARNING_RATE
        batch_size = batch_size or config.BATCH_SIZE

        input_size = features.shape[1]

        # Get or create model
        self.model = self.model_manager.get_or_create_model(
            self.ticker, input_size, self.hidden_layers
        )
        self.model = self.model.to(self.device)
        self.model.train()

        # Prepare data loaders
        X_tensor = torch.FloatTensor(features).to(self.device)
        y_tensor = torch.FloatTensor(targets).to(self.device)

        # Split into train/validation (90/10)
        split_idx = int(len(X_tensor) * 0.9)
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            epoch_train_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                optimizer.step()
                epoch_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_train_loss / max(num_batches, 1)

            # --- Validation ---
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = criterion(val_predictions, y_val).item()

            scheduler.step(val_loss)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )

        # Compute final detailed metrics
        self.model.eval()
        with torch.no_grad():
            all_preds = self.model(X_tensor).cpu().numpy()
            all_targets = y_tensor.cpu().numpy()

        horizon_names = list(config.PREDICTION_HORIZONS.keys())
        per_horizon_mae = {}
        for i, name in enumerate(horizon_names):
            mae = np.mean(np.abs(all_preds[:, i] - all_targets[:, i]))
            per_horizon_mae[name] = float(mae)

        training_info = {
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "best_val_loss": best_val_loss,
            "epochs_run": epochs,
            "num_samples": len(features),
            "per_horizon_mae": per_horizon_mae,
        }

        # Save model weights
        feature_names = FEATURE_COLUMNS.copy()
        feature_names.append("sentiment")
        self.model_manager.save_model(
            self.model, self.ticker, feature_names, training_info
        )

        logger.info(f"Training complete for {self.ticker}")
        logger.info(f"  Final train loss: {training_info['final_train_loss']:.6f}")
        logger.info(f"  Final val loss:   {training_info['final_val_loss']:.6f}")
        logger.info(f"  Per-horizon MAE:  {per_horizon_mae}")

        return training_info

    def run_full_training(self, start_year: int, start_month: int,
                          end_year: int, end_month: int,
                          use_rest: bool = False,
                          epochs: int = None) -> dict:
        """
        Run the complete training pipeline:
        1. Fetch data (flat files or REST)
        2. Get sentiment score
        3. Prepare samples
        4. Train

        Args:
            start_year: Start year of historical data
            start_month: Start month
            end_year: End year
            end_month: End month
            use_rest: If True, use REST API instead of flat files
            epochs: Override number of epochs

        Returns:
            Training info dict, or None on failure
        """
        # Step 1: Fetch data
        if use_rest:
            start_date = f"{start_year:04d}-{start_month:02d}-01"
            end_date = f"{end_year:04d}-{end_month:02d}-28"
            df = self.fetch_training_data_rest(start_date, end_date)
        else:
            df = self.fetch_training_data(
                start_year, start_month, end_year, end_month
            )

        if df is None or df.empty:
            logger.error("Failed to fetch training data")
            return None

        logger.info(f"Fetched {len(df)} rows of data for {self.ticker}")

        # Step 2: Get sentiment score
        sentiment_score = 0.0
        if self.use_sentiment:
            try:
                news = self.fetcher.fetch_ticker_news(self.ticker, limit=20)
                sentiment_score = self.sentiment_evaluator.evaluate_sentiment(
                    self.ticker, news
                )
                logger.info(f"Sentiment score for {self.ticker}: {sentiment_score}")
            except Exception as e:
                logger.warning(f"Sentiment evaluation failed: {e}")

        # Step 3: Prepare training samples
        features, targets = self.prepare_training_samples(df, sentiment_score)
        if features is None:
            logger.error("Failed to prepare training samples")
            return None

        # Step 4: Train
        return self.train(features, targets, epochs=epochs)


def run_incremental_training(ticker: str, start_year: int, start_month: int,
                             end_year: int, end_month: int,
                             hidden_layers: list[int] = None,
                             use_rest: bool = False,
                             use_sentiment: bool = True,
                             epochs_per_window: int = None):
    """
    Run training in 3-month incremental windows.

    For each 3-month window:
    - Uses data up to that point as training data
    - The next 3 months serve as the target/validation period
    - Model weights carry forward between windows

    Args:
        ticker: Stock ticker symbol
        start_year: Data start year
        start_month: Data start month
        end_year: Data end year
        end_month: Data end month
        hidden_layers: Hidden layer sizes (optional)
        use_rest: Use REST API instead of flat files
        use_sentiment: Enable sentiment analysis
        epochs_per_window: Epochs per 3-month training window

    Returns:
        List of training info dicts (one per window)
    """
    logger.info(
        f"Starting incremental training for {ticker} "
        f"({start_year}/{start_month} -> {end_year}/{end_month})"
    )

    trainer = StockTrainer(
        ticker, hidden_layers=hidden_layers, use_sentiment=use_sentiment
    )

    # First, fetch ALL the data
    if use_rest:
        start_date = f"{start_year:04d}-{start_month:02d}-01"
        end_date = f"{end_year:04d}-{end_month:02d}-28"
        full_df = trainer.fetch_training_data_rest(start_date, end_date)
    else:
        full_df = trainer.fetch_training_data(
            start_year, start_month, end_year, end_month
        )

    if full_df is None or full_df.empty:
        logger.error("No data available for training")
        return []

    logger.info(f"Total data points: {len(full_df)}")

    # Get sentiment (use same score for entire run)
    sentiment_score = 0.0
    if use_sentiment:
        try:
            news = trainer.fetcher.fetch_ticker_news(ticker, limit=20)
            sentiment_score = trainer.sentiment_evaluator.evaluate_sentiment(
                ticker, news
            )
        except Exception as e:
            logger.warning(f"Sentiment failed, using neutral: {e}")

    # Process in 3-month (63 trading days) windows
    window_size = config.TRAINING_WINDOW_DAYS  # 63 days
    max_horizon = max(config.PREDICTION_HORIZONS.values())  # 63 days

    all_results = []
    window_start = 0

    while window_start + window_size + max_horizon <= len(full_df):
        window_end = window_start + window_size + max_horizon
        window_df = full_df.iloc[:window_end].copy().reset_index(drop=True)

        logger.info(
            f"\n{'='*60}\n"
            f"Training window: rows {window_start} to {window_end} "
            f"(of {len(full_df)})\n"
            f"{'='*60}"
        )

        features, targets = trainer.prepare_training_samples(
            window_df, sentiment_score
        )

        if features is not None:
            result = trainer.train(
                features, targets, epochs=epochs_per_window
            )
            if result:
                result["window_start"] = window_start
                result["window_end"] = window_end
                all_results.append(result)

                # Print per-horizon results
                print(f"\nWindow {len(all_results)} results:")
                print(f"  Samples: {result['num_samples']}")
                print(f"  Val Loss: {result['final_val_loss']:.6f}")
                for h_name, mae in result["per_horizon_mae"].items():
                    print(f"  {h_name} MAE: {mae:.4f} ({mae*100:.2f}%)")

        # Advance by 3 months (63 trading days)
        window_start += window_size

    logger.info(f"\nIncremental training complete. {len(all_results)} windows trained.")
    return all_results
