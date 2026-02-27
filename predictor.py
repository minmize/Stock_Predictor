"""
Prediction mode module.

Uses the REST API to fetch the most recent 6 months of data (enough for
indicator warm-up + a full 63-day feature window), computes features with
sector encoding, runs the universal trained neural network, and outputs
predicted price movements for 6 time horizons.

Also evaluates current market sentiment via the Anthropic API.
"""

import logging
from datetime import datetime

import numpy as np
import torch

import config
from data_fetcher import MassiveDataFetcher, normalize_dataframe
from sentiment import SentimentEvaluator
from features import (
    build_feature_matrix,
    get_feature_count,
    compute_technical_features,
    FEATURE_COLUMNS,
)
from neural_net import ModelManager

logger = logging.getLogger(__name__)


class StockPredictor:
    """
    Makes predictions for a stock ticker using the universal trained model.

    Workflow:
    1. Fetches recent 6 months of daily data via REST API
       (6 months ensures enough data after indicator warm-up for
       a full 63-day feature window)
    2. Evaluates current sentiment via Anthropic API
    3. Builds feature vector with sector encoding
    4. Loads universal trained model
    5. Runs inference
    6. Outputs predictions
    """

    def __init__(self, ticker: str, sector: str = "other",
                 hidden_layers: list[int] = None,
                 use_sentiment: bool = True):
        """
        Args:
            ticker: Stock ticker symbol
            sector: Sector name for one-hot encoding
            hidden_layers: Hidden layer config (must match training)
            use_sentiment: Whether to use sentiment analysis
        """
        self.ticker = ticker
        self.sector = sector
        self.hidden_layers = hidden_layers or list(config.HIDDEN_LAYERS)
        self.use_sentiment = use_sentiment

        self.fetcher = MassiveDataFetcher()
        self.model_manager = ModelManager()
        self.sentiment_evaluator = SentimentEvaluator() if use_sentiment else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self) -> dict:
        """
        Run the full prediction pipeline.

        Returns:
            Dict with keys:
            - ticker: str
            - sector: str
            - current_price: float
            - predictions: dict mapping horizon name to predicted % change
            - sentiment_score: float
            - data_range: dict with start_date and end_date
            - timestamp: str
            Or None on failure.
        """
        # Step 1: Fetch recent data via REST API (6 months to ensure
        # enough clean rows after indicator warm-up NaN dropping)
        logger.info(f"Fetching recent 6 months of data for {self.ticker}")
        raw_df = self.fetcher.fetch_recent_data(self.ticker, months=6)

        if raw_df is None or raw_df.empty:
            logger.error(f"No recent data available for {self.ticker}")
            return None

        df = normalize_dataframe(raw_df)
        logger.info(f"Got {len(df)} days of data for {self.ticker}")

        current_price = float(df.iloc[-1]["close"])
        data_start = df.iloc[0]["date"].strftime("%Y-%m-%d")
        data_end = df.iloc[-1]["date"].strftime("%Y-%m-%d")

        # Step 2: Get sentiment score
        sentiment_score = 0.0
        if self.use_sentiment:
            try:
                news = self.fetcher.fetch_ticker_news(self.ticker, limit=20)
                sentiment_score = self.sentiment_evaluator.evaluate_sentiment(
                    self.ticker, news
                )
                logger.info(
                    f"Current sentiment for {self.ticker}: {sentiment_score}"
                )
            except Exception as e:
                logger.warning(f"Sentiment evaluation failed: {e}")

        # Step 3: Build feature vector (with sector encoding)
        feature_vector = build_feature_matrix(
            df, sentiment_score=sentiment_score,
            sector=self.sector,
            lookback_days=config.TRAINING_WINDOW_DAYS
        )

        if feature_vector is None or len(feature_vector) == 0:
            logger.error("Failed to build feature vector")
            return None

        input_size = len(feature_vector)
        expected_size = get_feature_count()
        logger.info(
            f"Feature vector size: {input_size} (expected: {expected_size})"
        )

        if input_size != expected_size:
            logger.error(
                f"Feature vector size mismatch: got {input_size}, "
                f"expected {expected_size}. This usually means not enough "
                f"historical data was fetched. Need at least "
                f"{config.TRAINING_WINDOW_DAYS + 30} trading days of raw data."
            )
            return None

        # Step 4: Load universal trained model
        model = self.model_manager.load_model(
            input_size=input_size, hidden_layers=self.hidden_layers
        )

        if model is None:
            logger.error(
                "No trained universal model found. "
                "Run training mode first with any stock."
            )
            return None

        model = model.to(self.device)
        model.eval()

        # Step 5: Run inference
        with torch.no_grad():
            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            output = model(input_tensor)
            predictions_raw = output.squeeze(0).cpu().numpy()

        # Step 6: Format predictions
        horizon_names = list(config.PREDICTION_HORIZONS.keys())
        predictions = {}
        for i, name in enumerate(horizon_names):
            pct_change = float(predictions_raw[i]) * 100  # Convert to percentage
            predicted_price = current_price * (1 + predictions_raw[i])
            predictions[name] = {
                "percent_change": round(pct_change, 2),
                "predicted_price": round(predicted_price, 2),
                "horizon_days": config.PREDICTION_HORIZONS[name],
            }

        result = {
            "ticker": self.ticker,
            "sector": self.sector,
            "current_price": round(current_price, 2),
            "predictions": predictions,
            "sentiment_score": round(sentiment_score, 3),
            "data_range": {
                "start_date": data_start,
                "end_date": data_end,
                "num_days": len(df),
            },
            "timestamp": datetime.now().isoformat(),
            "model_info": model.get_architecture_info(),
        }

        return result


def format_prediction_report(result: dict) -> str:
    """
    Format a prediction result dict into a human-readable report.

    Args:
        result: Dict returned by StockPredictor.predict()

    Returns:
        Formatted string report
    """
    if result is None:
        return "Prediction failed. No results available."

    lines = []
    lines.append("=" * 60)
    lines.append(f"  STOCK PREDICTION REPORT - {result['ticker']}")
    lines.append("=" * 60)
    lines.append(f"  Generated: {result['timestamp']}")
    lines.append(f"  Sector: {result.get('sector', 'unknown')}")
    lines.append(
        f"  Data range: {result['data_range']['start_date']} to "
        f"{result['data_range']['end_date']} "
        f"({result['data_range']['num_days']} trading days)"
    )
    lines.append(f"  Current price: ${result['current_price']:.2f}")
    lines.append(
        f"  Sentiment score: {result['sentiment_score']:+.3f} "
        f"({'Positive' if result['sentiment_score'] > 0.1 else 'Negative' if result['sentiment_score'] < -0.1 else 'Neutral'})"
    )
    lines.append("")
    lines.append("  Predictions:")
    lines.append("  " + "-" * 56)
    lines.append(
        f"  {'Horizon':<15} {'% Change':>10} {'Price':>12} {'Direction':>12}"
    )
    lines.append("  " + "-" * 56)

    horizon_labels = {
        "1_day": "1 Day",
        "4_days": "4 Days",
        "1_week": "1 Week",
        "2_weeks": "2 Weeks",
        "1_month": "1 Month",
        "3_months": "3 Months",
    }

    for name, pred in result["predictions"].items():
        pct = pred["percent_change"]
        price = pred["predicted_price"]
        direction = "UP" if pct > 0.5 else "DOWN" if pct < -0.5 else "FLAT"
        sign = "+" if pct >= 0 else ""
        label = horizon_labels.get(name, name)
        lines.append(
            f"  {label:<15} {sign}{pct:>9.2f}% ${price:>10.2f} {direction:>12}"
        )

    lines.append("  " + "-" * 56)
    lines.append("")

    # Model info
    info = result.get("model_info", {})
    if info:
        lines.append(f"  Model: {info.get('total_params', '?')} parameters")
        lines.append(
            f"  Architecture: {info.get('input_size', '?')} -> "
            f"{info.get('hidden_layers', '?')} -> {info.get('num_outputs', '?')}"
        )

    lines.append("=" * 60)
    lines.append(
        "  DISCLAIMER: These predictions are for educational purposes"
    )
    lines.append(
        "  only and should NOT be used as financial advice."
    )
    lines.append("=" * 60)

    return "\n".join(lines)
