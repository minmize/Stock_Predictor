"""
Training mode module.

Training process:
1. Splits date range into 3-month batches (working backwards from end date)
2. Discards any leftover time that doesn't fill a 3-month batch
3. Fetches ALL data in one REST call (with buffers for indicator warm-up
   at the start and forward targets at the end)
4. Computes technical features once on the full dataset
5. For each 3-month batch:
   a. Selects anchor days that fall within the batch date range
   b. Builds training samples using pre-computed features (63-day lookback)
      and forward targets (up to 63 days ahead) from the full dataset
   c. Trains the universal neural network
   d. Weights carry forward between batches
6. Saves universal weights to disk after each batch

The model uses universal weights shared across all stocks. Sector
information is encoded as a single normalized value in the input features.
Default training range is 4 years back from the current date.
"""

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
from sentiment import SentimentEvaluator
from features import (
    compute_targets,
    encode_sector,
    FEATURE_COLUMNS,
    compute_technical_features,
)
from neural_net import ModelManager

logger = logging.getLogger(__name__)


def compute_3month_batches(end_date: datetime, start_date: datetime) -> list[tuple]:
    """
    Split a date range into 3-month batches, working backwards from end_date.
    Discards any leftover time at the beginning that doesn't fill a full batch.

    Args:
        end_date: The latest date in the range
        start_date: The earliest date in the range

    Returns:
        List of (batch_start, batch_end) datetime tuples, ordered
        chronologically (earliest first).
    """
    batches = []
    cursor = end_date

    while True:
        batch_start = cursor - timedelta(days=91)  # ~3 calendar months
        if batch_start < start_date:
            # Remaining time doesn't fill a 3-month batch; discard it
            break
        batches.append((batch_start, cursor))
        cursor = batch_start

    # Reverse so batches are chronological (train earliest first)
    batches.reverse()
    return batches


class StockTrainer:
    """
    Handles the full training pipeline for a stock ticker.

    Coordinates data fetching, feature engineering, training loop,
    and universal model persistence. The same model weights are used
    for all stocks, with sector information encoded in the input.

    Training always proceeds in 3-month batches. If no dates are
    specified, defaults to 4 years back from today.
    """

    def __init__(self, ticker: str, sector: str = "other",
                 hidden_layers: list[int] = None,
                 use_sentiment: bool = True):
        """
        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")
            sector: Sector name (e.g. "technology"), auto-detected if not given
            hidden_layers: Optional override for hidden layer sizes
            use_sentiment: Whether to fetch and use sentiment scores
        """
        self.ticker = ticker
        self.sector = sector
        self.hidden_layers = hidden_layers or list(config.HIDDEN_LAYERS)
        self.use_sentiment = use_sentiment

        self.fetcher = MassiveDataFetcher()
        self.model_manager = ModelManager()
        self.sentiment_evaluator = SentimentEvaluator() if use_sentiment else None

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fetch_training_data(self, start_date: str,
                            end_date: str) -> Optional[any]:
        """
        Fetch historical data for training via REST API.

        Args:
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"

        Returns:
            Normalized pandas DataFrame, or None on failure
        """
        df = self.fetcher.fetch_rest_aggregates(
            self.ticker, start_date, end_date
        )
        if df.empty:
            return None
        return normalize_dataframe(df)

    def train_batch(self, features: np.ndarray, targets: np.ndarray,
                    epochs: int = None, learning_rate: float = None,
                    batch_size: int = None) -> dict:
        """
        Run the training loop for one batch of data.

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

        # Get or create universal model
        self.model = self.model_manager.get_or_create_model(
            input_size, self.hidden_layers
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
            "ticker": self.ticker,
            "sector": self.sector,
        }

        # Save universal model weights after each batch
        feature_names = FEATURE_COLUMNS.copy()
        feature_names.append("sector")
        feature_names.append("sentiment")
        self.model_manager.save_model(
            self.model, feature_names, training_info
        )

        logger.info(f"Batch training complete for {self.ticker}")
        logger.info(f"  Final train loss: {training_info['final_train_loss']:.6f}")
        logger.info(f"  Final val loss:   {training_info['final_val_loss']:.6f}")

        return training_info

    def run_training(self, end_date: datetime = None,
                     start_date: datetime = None,
                     epochs_per_batch: int = None) -> list[dict]:
        """
        Run the complete training pipeline in 3-month batches.

        Data flow:
        1. Compute the "trainable" anchor range: start_date to
           (end_date - max_horizon), since anchor days need forward
           data to compute target prices.
        2. Split the anchor range into 3-month batches (backwards
           from the end, earliest-first). Leftover time at the start
           that doesn't fill a full batch is discarded.
        3. Fetch ALL data in one API call: from (first_batch_start
           - warmup - lookback) to end_date.
        4. Compute technical features once on the full dataset.
        5. For each batch, select anchor days within its date range,
           build samples using the pre-computed features (63-day
           lookback) and forward targets (up to 63 days ahead).
        6. Train on each batch; weights carry forward.

        Args:
            end_date: Latest date for training data (default: today)
            start_date: Earliest date (default: 4 years before end_date)
            epochs_per_batch: Epochs per 3-month batch

        Returns:
            List of training info dicts (one per batch)
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=4 * 365)

        window = config.TRAINING_WINDOW_DAYS
        max_horizon = max(config.PREDICTION_HORIZONS.values())

        # ----------------------------------------------------------
        # Anchor days need max_horizon days of FUTURE data for targets.
        # So the last valid anchor day is ~max_horizon trading days
        # before end_date. Reserve that tail for forward targets only.
        # ----------------------------------------------------------
        target_reserve = timedelta(days=int(max_horizon * 1.5))  # calendar days
        anchor_end = end_date - target_reserve

        # Compute 3-month batches over the trainable anchor range
        batches = compute_3month_batches(anchor_end, start_date)

        if not batches:
            logger.error(
                "Date range too short for any 3-month batches. "
                f"Need at least ~6 months for one batch + forward targets. "
                f"Range: {start_date.date()} to {end_date.date()}"
            )
            return []

        discarded_days = (batches[0][0] - start_date).days
        if discarded_days > 0:
            logger.info(
                f"Discarding {discarded_days} days at the start that "
                f"don't fill a 3-month batch"
            )

        logger.info(
            f"Training {self.ticker} (sector: {self.sector}) in "
            f"{len(batches)} batches of 3 months"
        )
        logger.info(
            f"  Anchor range: {batches[0][0].date()} to "
            f"{batches[-1][1].date()}"
        )
        logger.info(
            f"  Forward target data extends to: {end_date.date()}"
        )

        # ----------------------------------------------------------
        # Fetch ALL data in one shot:
        #   - Before first batch: warmup (~30 rows) + lookback (63 rows)
        #   - After last batch: forward targets up to end_date
        # ----------------------------------------------------------
        warmup_lookback = timedelta(days=int((30 + window) * 1.6))
        fetch_start = (batches[0][0] - warmup_lookback).strftime("%Y-%m-%d")
        fetch_end = end_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching full dataset: {fetch_start} to {fetch_end}")
        df = self.fetch_training_data(fetch_start, fetch_end)
        if df is None or df.empty:
            logger.error("Failed to fetch training data")
            return []

        logger.info(f"Total rows fetched: {len(df)}")

        # Compute technical features once on the full dataset
        feat_df = compute_technical_features(df)
        feat_df = feat_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        logger.info(f"Clean rows after indicator warm-up: {len(feat_df)}")

        if len(feat_df) < window + max_horizon:
            logger.error(
                f"Not enough clean data: {len(feat_df)} rows, "
                f"need at least {window + max_horizon}. "
                f"Try a longer date range or a ticker with more history."
            )
            return []

        # Log the valid training range for diagnostics
        first_valid_idx = window
        last_valid_idx = len(feat_df) - max_horizon - 1
        logger.info(
            f"Valid anchor range in data: index {first_valid_idx} "
            f"to {last_valid_idx} "
            f"({last_valid_idx - first_valid_idx + 1} positions, "
            f"dates {feat_df.iloc[first_valid_idx]['date'].date()} "
            f"to {feat_df.iloc[last_valid_idx]['date'].date()})"
        )

        # Get sentiment once for the entire run
        sentiment_score = 0.0
        if self.use_sentiment:
            try:
                news = self.fetcher.fetch_ticker_news(self.ticker, limit=20)
                sentiment_score = self.sentiment_evaluator.evaluate_sentiment(
                    self.ticker, news
                )
                logger.info(
                    f"Sentiment score for {self.ticker}: {sentiment_score}"
                )
            except Exception as e:
                logger.warning(f"Sentiment evaluation failed: {e}")

        # Pre-compute values used for all batches
        sector_value = encode_sector(self.sector)

        all_results = []
        total_samples = 0

        for batch_num, (batch_start, batch_end) in enumerate(batches, 1):
            start_str = batch_start.strftime("%Y-%m-%d")
            end_str = batch_end.strftime("%Y-%m-%d")

            print(
                f"\n{'='*60}\n"
                f"  Batch {batch_num}/{len(batches)}: "
                f"{start_str} to {end_str}\n"
                f"{'='*60}"
            )

            # Find rows whose date falls within this batch's range.
            # These are the "anchor" days we build training samples around.
            batch_mask = (
                (feat_df["date"] >= batch_start)
                & (feat_df["date"] <= batch_end)
            )
            batch_indices = feat_df.index[batch_mask].tolist()

            # Keep only positions with enough lookback AND forward data
            valid_indices = [
                i for i in batch_indices
                if i >= window and i + max_horizon < len(feat_df)
            ]

            if not valid_indices:
                logger.warning(
                    f"Batch {batch_num}: {len(batch_indices)} days in range, "
                    f"0 valid anchor positions — skipping"
                )
                continue

            logger.info(
                f"Batch {batch_num}: {len(batch_indices)} days in range, "
                f"{len(valid_indices)} valid anchor positions"
            )

            # Build feature vectors and targets for this batch
            features_list = []
            targets_list = []

            for i in valid_indices:
                # 63-day lookback window of pre-computed features
                window_data = feat_df.iloc[i - window:i][FEATURE_COLUMNS].values
                window_data = np.nan_to_num(
                    window_data, nan=0.0, posinf=1.0, neginf=-1.0
                )

                flat = window_data.flatten()
                flat = np.append(flat, sector_value)
                flat = np.append(flat, sentiment_score)

                # Forward-looking targets
                target = compute_targets(feat_df, i)
                if target is not None:
                    features_list.append(flat)
                    targets_list.append(target)

            if not features_list:
                logger.warning(
                    f"Batch {batch_num}: no valid samples — skipping"
                )
                continue

            features = np.array(features_list, dtype=np.float32)
            targets = np.array(targets_list, dtype=np.float32)
            total_samples += len(features)

            logger.info(
                f"Batch {batch_num}: {len(features)} training samples, "
                f"feature dim = {features.shape[1]}"
            )

            # Train on this batch (weights carry forward)
            result = self.train_batch(
                features, targets, epochs=epochs_per_batch
            )
            if result:
                result["batch_num"] = batch_num
                result["batch_start"] = start_str
                result["batch_end"] = end_str
                all_results.append(result)

                print(f"  Samples: {result['num_samples']}")
                print(f"  Val Loss: {result['final_val_loss']:.6f}")
                for h_name, mae in result["per_horizon_mae"].items():
                    print(f"  {h_name} MAE: {mae:.4f} ({mae*100:.2f}%)")

        print(
            f"\nTraining complete. {len(all_results)}/{len(batches)} "
            f"batches processed, {total_samples} total samples."
        )
        logger.info(
            f"Training complete. {len(all_results)}/{len(batches)} "
            f"batches processed, {total_samples} total samples."
        )
        return all_results
