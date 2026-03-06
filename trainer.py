"""
Training mode module.

Training process:
1. Splits date range into 3-month batches (working backwards from end date)
2. Discards any leftover time that doesn't fill a 3-month batch
3. Fetches ALL price data in one REST call (with buffers for indicator
   warm-up at the start and forward targets at the end)
4. Computes technical features once on the full dataset
5. For every batch, fetches fundamental data filed before the batch
   end date (no leakage). Every 3 batches, also refreshes:
   b. Ticker sentiment from 10 historical news articles
   c. World events sentiment from 10 historical market articles
6. For each 3-month batch:
   a. Selects anchor days that fall within the batch date range
   b. Builds training samples using pre-computed features (63-day lookback),
      historical fundamentals, historical sentiment, and forward targets
   c. Trains the universal neural network
   d. Weights carry forward between batches
7. Saves universal weights + optimizer state to disk after each batch

The model uses universal weights shared across all stocks. Sector
information is encoded as a single normalized value in the input features.
Default training range is 4 years back from the current date.

Prediction mode uses current data: 50 news articles, latest financials.
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
    normalize_fundamentals,
    FEATURE_COLUMNS,
    FUNDAMENTAL_KEYS,
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

    def _init_model_and_optimizer(self, input_size: int,
                                  learning_rate: float = None):
        """
        Load or create the universal model and optimizer ONCE.

        If a compatible saved model exists, loads it and restores
        the optimizer state so that Adam's momentum and adaptive
        learning rates carry forward across training runs.

        Sets self.model and self.optimizer.

        Args:
            input_size: Number of input features
            learning_rate: Learning rate for the optimizer
        """
        learning_rate = learning_rate or config.LEARNING_RATE

        # Load or create model
        self.model = self.model_manager.get_or_create_model(
            input_size, self.hidden_layers
        )
        self.model = self.model.to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Restore saved optimizer state if available (carries momentum
        # and adaptive learning rates from prior training runs)
        saved_opt_state = self.model_manager.get_saved_optimizer_state()
        if saved_opt_state is not None:
            try:
                self.optimizer.load_state_dict(saved_opt_state)
                # Move optimizer state to device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                # Reset LR to configured base value. The saved LR may have
                # been reduced by overshoot detection at the end of a previous
                # session; keeping that stale reduced LR would make the network
                # learn far too slowly from the start of a new run.
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = learning_rate
                logger.info(
                    "Restored optimizer state from previous training run "
                    f"(LR reset to {learning_rate})"
                )
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Could not restore optimizer state (model architecture "
                    f"may have changed): {e}. Starting fresh optimizer."
                )

        # NOTE: LR management (overshoot detection + reduction) is handled
        # inside each train_batch call via consecutive_bad epoch counting.
        # The saved optimizer LR is always reset to the base value above so
        # each run starts at the configured rate, not a stale reduced value.

        logger.info(
            f"Model ready: {sum(p.numel() for p in self.model.parameters())} "
            f"parameters, lr={learning_rate}"
        )

    def train_batch(self, features: np.ndarray, targets: np.ndarray,
                    close_prices: np.ndarray = None,
                    epochs: int = None,
                    batch_size: int = None) -> dict:
        """
        Run the training loop for one batch of data.

        Uses self.model and self.optimizer which must be initialized
        by _init_model_and_optimizer() before calling this method.
        The optimizer state (Adam momentum / per-param LRs) persists
        across batch calls; the LR scheduler is created fresh each
        batch so its patience counter never accumulates across the
        full training run.

        Loss function: price-based MSE. For each sample the model's
        predicted % change is converted back to an absolute price
        (anchor_close × (1 + pred)) and compared to the actual future
        close price (anchor_close × (1 + target)). This gives the
        network a more natural, dollar-scale error signal.

        Best-epoch checkpointing: the model state at the epoch with
        the lowest validation loss is restored before saving so that
        we never persist an overfitting checkpoint.

        Args:
            features:     Feature matrix  (num_samples, input_size)
            targets:      Target matrix   (num_samples, 6)  — % change fractions
            close_prices: Anchor-day close prices (num_samples,) for
                          price-based loss. Falls back to plain MSE if None.
            epochs:       Number of training epochs
            batch_size:   Mini-batch size

        Returns:
            Dict with training statistics
        """
        epochs = epochs or config.EPOCHS_PER_WINDOW
        batch_size = batch_size or config.BATCH_SIZE

        self.model.train()

        # Prepare tensors
        X_tensor = torch.FloatTensor(features).to(self.device)
        y_tensor = torch.FloatTensor(targets).to(self.device)
        use_price_loss = close_prices is not None
        if use_price_loss:
            c_tensor = torch.FloatTensor(close_prices).to(self.device)  # (N,)

        # Split into train/validation (90/10, chronological)
        split_idx = int(len(X_tensor) * 0.9)
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        if use_price_loss:
            c_train = c_tensor[:split_idx]
            c_val   = c_tensor[split_idx:]
            train_dataset = TensorDataset(X_train, y_train, c_train)
        else:
            train_dataset = TensorDataset(X_train, y_train)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        criterion = nn.MSELoss()

        def compute_loss(preds, tgts, close_vals=None):
            """
            Price-based MSE: reconstruct predicted and actual prices
            from the anchor close, then compute MSE in price space.

            pred_price  = close × (1 + model_output)
            actual_price = close × (1 + target_pct)
            loss = MSE(pred_price, actual_price)

            Falls back to plain % change MSE when close_vals is None.
            """
            if close_vals is not None:
                c = close_vals.unsqueeze(1)          # (batch, 1) → broadcasts over 6 horizons
                pred_price   = c * (1.0 + preds)
                actual_price = c * (1.0 + tgts)
                return criterion(pred_price, actual_price)
            return criterion(preds, tgts)

        # Learning rate is set proportionally to RMSE after each epoch:
        #   lr = clamp(LR_SCALE × RMSE, MIN_LR, MAX_LR)
        # High loss → high LR; as the model improves LR decays automatically.
        # Overshoot-recovery parameters.
        # When val_loss has risen for _OVERSHOOT_PATIENCE consecutive epochs
        # the model has passed the optimum.  We restore the last best weights,
        # dampen Adam's first moment (exp_avg) so the accumulated momentum
        # that carried us past the optimum can't push us past it again, then
        # halve the LR so subsequent steps are smaller.
        _OVERSHOOT_PATIENCE = 5    # epochs of rising val_loss before acting
        _LR_DECAY_FACTOR    = 0.5  # LR multiplier on each detected overshoot
        _MIN_LR             = 1e-6  # hard floor

        history = {"train_loss": [], "val_loss": []}
        best_val_loss    = float("inf")
        best_model_state = None   # snapshot at best validation epoch
        consecutive_bad  = 0      # epochs since last val_loss improvement

        for epoch in range(epochs):
            # --- Training pass ---
            self.model.train()
            epoch_train_loss = 0.0
            num_batches = 0

            for batch_data in train_loader:
                if use_price_loss:
                    batch_X, batch_y, batch_c = batch_data
                else:
                    batch_X, batch_y = batch_data
                    batch_c = None

                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = compute_loss(predictions, batch_y, batch_c)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                self.optimizer.step()
                epoch_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_train_loss / max(num_batches, 1)

            # --- Validation pass ---
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = compute_loss(
                    val_predictions, y_val,
                    c_val if use_price_loss else None
                ).item()

            # Proportional LR: RMSE (in price units) drives the learning rate
            rmse = val_loss ** 0.5
            new_lr = max(config.MIN_LR, min(config.MAX_LR, config.LR_SCALE * rmse))
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            # Overshoot recovery ─────────────────────────────────────────────
            # On improvement: snapshot weights and reset the bad-epoch counter.
            # On sustained divergence: restore the best-known weights so we
            # step back to the pre-overshoot point, then dampen Adam's first
            # moment (exp_avg) so the momentum that pushed us past the optimum
            # can't repeat the error, and reduce the LR so the next approach
            # to the optimum is made with smaller steps.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.clone()
                    for k, v in self.model.state_dict().items()
                }
                consecutive_bad = 0
            else:
                consecutive_bad += 1
                if consecutive_bad >= _OVERSHOOT_PATIENCE \
                        and best_model_state is not None:
                    # Step back to the last known-good weights
                    self.model.load_state_dict(best_model_state)
                    # Dampen first moment so accumulated momentum can't push
                    # past the optimum again on the next forward pass
                    for state in self.optimizer.state.values():
                        if isinstance(state, dict) and "exp_avg" in state:
                            state["exp_avg"].mul_(0.1)
                    # Reduce LR (floor at _MIN_LR)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = max(pg["lr"] * _LR_DECAY_FACTOR, _MIN_LR)
                    logger.info(
                        f"Epoch {epoch + 1}: overshoot detected — restored "
                        f"best weights, dampened momentum, "
                        f"LR → {self.optimizer.param_groups[0]['lr']:.2e}"
                    )
                    consecutive_bad = 0
            # ────────────────────────────────────────────────────────────────

            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"LR: {current_lr:.2e}"
                )

        # Restore best-epoch weights before saving so we never persist
        # an overfit checkpoint
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(
                f"Restored best-epoch weights "
                f"(val_loss={best_val_loss:.4f})"
            )

        # Compute final metrics on % change fractions for interpretability
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

        # Save best-epoch model + optimizer state for crash recovery
        # and cross-run persistence
        feature_names = FEATURE_COLUMNS.copy()
        feature_names.append("sector")
        feature_names.append("ticker_sentiment")
        feature_names.append("world_sentiment")
        feature_names.extend(FUNDAMENTAL_KEYS)
        self.model_manager.save_model(
            self.model, feature_names, training_info,
            optimizer=self.optimizer
        )

        logger.info(f"Batch training complete for {self.ticker}")
        logger.info(f"  Best val loss:    {training_info['best_val_loss']:.4f}")
        logger.info(f"  Final train loss: {training_info['final_train_loss']:.4f}")
        logger.info(f"  Final val loss:   {training_info['final_val_loss']:.4f}")

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

        # ----------------------------------------------------------
        # Initialize model + optimizer ONCE before all batches.
        # Input = (window * technical_features) + sector + ticker_sentiment
        #         + world_sentiment + fundamental_metrics
        # ----------------------------------------------------------
        input_size = (
            (window * len(FEATURE_COLUMNS))
            + 1   # sector
            + 1   # ticker sentiment
            + 1   # world events sentiment
            + len(FUNDAMENTAL_KEYS)  # fundamental metrics
        )
        self._init_model_and_optimizer(input_size)

        # Pre-compute values used for all batches
        sector_value = encode_sector(self.sector)

        # Sentiment and fundamentals are refreshed every 3 batches
        # using historical data from the batch's time period to
        # avoid data leakage (no future information).
        sentiment_score = 0.0
        world_sentiment_score = 0.0
        fundamental_values = normalize_fundamentals({})
        _SENTIMENT_REFRESH_INTERVAL = 3

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

            # -------------------------------------------------------
            # Fundamentals: fetch every batch using filings available
            # AS OF the batch end date (no leakage).
            # -------------------------------------------------------
            try:
                fundamentals = self.fetcher.fetch_financials(
                    self.ticker, as_of_date=end_str
                )
                fundamental_values = normalize_fundamentals(fundamentals)
                logger.info(
                    f"Fetched fundamentals for {self.ticker} as of {end_str}"
                )
            except Exception as e:
                logger.warning(
                    f"Fundamentals fetch for {end_str} failed: {e}"
                )

            # -------------------------------------------------------
            # Sentiment: refresh every N batches (Claude API calls are
            # expensive). Uses historical news published before batch
            # end date to avoid data leakage.
            # -------------------------------------------------------
            if (batch_num - 1) % _SENTIMENT_REFRESH_INTERVAL == 0:
                if self.use_sentiment:
                    # Ticker news published before batch end
                    try:
                        news = self.fetcher.fetch_ticker_news_historical(
                            self.ticker, before_date=end_str, limit=10
                        )
                        sentiment_score = (
                            self.sentiment_evaluator.evaluate_sentiment(
                                self.ticker, news
                            )
                        )
                        logger.info(
                            f"Historical sentiment for {self.ticker} "
                            f"as of {end_str}: {sentiment_score}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Historical sentiment failed: {e}"
                        )

                    # World news published before batch end
                    try:
                        world_news = (
                            self.fetcher.fetch_market_news_historical(
                                before_date=end_str, limit=10
                            )
                        )
                        world_sentiment_score = (
                            self.sentiment_evaluator.evaluate_world_events(
                                self.ticker, self.sector, world_news
                            )
                        )
                        logger.info(
                            f"Historical world sentiment as of "
                            f"{end_str}: {world_sentiment_score}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Historical world sentiment failed: {e}"
                        )

                logger.info(
                    f"Refreshed sentiment for batch {batch_num}: "
                    f"sentiment={sentiment_score:.3f}, "
                    f"world={world_sentiment_score:.3f}"
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

            # Build feature vectors, targets, and anchor close prices
            features_list = []
            targets_list = []
            close_prices_list = []

            for i in valid_indices:
                # 63-day lookback window of pre-computed features
                window_data = feat_df.iloc[i - window:i][FEATURE_COLUMNS].values
                window_data = np.nan_to_num(
                    window_data, nan=0.0, posinf=1.0, neginf=-1.0
                )

                flat = window_data.flatten()
                flat = np.append(flat, sector_value)
                flat = np.append(flat, sentiment_score)
                flat = np.append(flat, world_sentiment_score)
                flat = np.append(flat, fundamental_values)

                # Forward-looking targets
                target = compute_targets(feat_df, i)
                if target is not None:
                    features_list.append(flat)
                    targets_list.append(target)
                    # Anchor close: used to reconstruct predicted prices
                    # in the loss function (price-based MSE)
                    close_prices_list.append(
                        float(feat_df.iloc[i]["close"])
                    )

            if not features_list:
                logger.warning(
                    f"Batch {batch_num}: no valid samples — skipping"
                )
                continue

            features = np.array(features_list, dtype=np.float32)
            targets = np.array(targets_list, dtype=np.float32)
            close_prices = np.array(close_prices_list, dtype=np.float32)
            total_samples += len(features)

            logger.info(
                f"Batch {batch_num}: {len(features)} training samples, "
                f"feature dim = {features.shape[1]}"
            )

            # Train on this batch (weights carry forward)
            result = self.train_batch(
                features, targets,
                close_prices=close_prices,
                epochs=epochs_per_batch,
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
