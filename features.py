"""
Feature engineering module.

Takes raw OHLCV data and produces feature vectors for the neural network.
Each data point in the input window is turned into features, and the
neural network input layer is automatically sized to match.

Features computed per day (27 indicators):
- Daily returns (close-to-close % change, log returns)
- Intraday range (high - low) / close
- Gap (open vs previous close)
- Volume change ratio
- Simple moving averages (5, 10, 21 day)
- Exponential moving averages (12, 26 day)
- MACD signal
- Bollinger Band position
- RSI (14-day)
- Volatility (rolling std of returns, 5 and 21 day)
- Average True Range (14-day)
- Close position within daily range
- VWAP ratio, transaction intensity ratio
- Rate of Change (5, 10 day momentum)
- Stochastic Oscillator (%K, %D)
- Williams %R
- On-Balance Volume ratio
- Money Flow Index (14-day)
- Chaikin Money Flow (20-day)
- Sector value (normalized index into standard sector array, 0 to 1)
- Sentiment score (from Anthropic API)
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features from OHLCV data.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
            Optionally: transactions, vwap

    Returns:
        DataFrame with additional feature columns added
    """
    feat = df.copy()

    # Daily returns
    feat["return"] = feat["close"].pct_change()

    # Log returns (more normally distributed)
    feat["log_return"] = np.log(feat["close"] / feat["close"].shift(1))

    # Intraday range normalized by close
    feat["intraday_range"] = (feat["high"] - feat["low"]) / feat["close"]

    # Gap: open vs previous close
    feat["gap"] = (feat["open"] - feat["close"].shift(1)) / feat["close"].shift(1)

    # Volume change ratio
    feat["volume_ratio"] = feat["volume"] / feat["volume"].rolling(5).mean()

    # Simple moving averages (relative to current close)
    for window in [5, 10, 21]:
        sma = feat["close"].rolling(window).mean()
        feat[f"sma_{window}_ratio"] = feat["close"] / sma

    # Exponential moving averages
    for span in [12, 26]:
        ema = feat["close"].ewm(span=span).mean()
        feat[f"ema_{span}_ratio"] = feat["close"] / ema

    # MACD
    ema_12 = feat["close"].ewm(span=12).mean()
    ema_26 = feat["close"].ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9).mean()
    feat["macd"] = (macd_line - signal_line) / feat["close"]

    # Bollinger Band position
    sma_20 = feat["close"].rolling(20).mean()
    std_20 = feat["close"].rolling(20).std()
    feat["bollinger_pos"] = (feat["close"] - sma_20) / (2 * std_20 + 1e-10)

    # RSI (14-day)
    delta = feat["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feat["rsi"] = (100 - (100 / (1 + rs))) / 100  # Normalize to [0, 1]

    # Volatility (rolling standard deviation of returns)
    feat["volatility_5"] = feat["return"].rolling(5).std()
    feat["volatility_21"] = feat["return"].rolling(21).std()

    # Average True Range (normalized)
    high_low = feat["high"] - feat["low"]
    high_close = (feat["high"] - feat["close"].shift(1)).abs()
    low_close = (feat["low"] - feat["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    feat["atr"] = true_range.rolling(14).mean() / feat["close"]

    # Price position within day range
    feat["close_position"] = (
        (feat["close"] - feat["low"]) / (feat["high"] - feat["low"] + 1e-10)
    )

    # Volume-weighted metrics (if vwap available)
    if "vwap" in feat.columns:
        feat["vwap_ratio"] = feat["close"] / (feat["vwap"] + 1e-10)
    else:
        feat["vwap_ratio"] = 1.0

    # Transaction intensity (if available)
    if "transactions" in feat.columns:
        feat["txn_ratio"] = (
            feat["transactions"]
            / (feat["transactions"].rolling(5).mean() + 1e-10)
        )
    else:
        feat["txn_ratio"] = 1.0

    # --- Additional momentum & volume indicators ---

    # Rate of Change (momentum)
    feat["roc_5"] = feat["close"].pct_change(5)
    feat["roc_10"] = feat["close"].pct_change(10)

    # Stochastic Oscillator (14-day)
    lowest_14 = feat["low"].rolling(14).min()
    highest_14 = feat["high"].rolling(14).max()
    feat["stoch_k"] = (
        (feat["close"] - lowest_14) / (highest_14 - lowest_14 + 1e-10)
    )
    feat["stoch_d"] = feat["stoch_k"].rolling(3).mean()

    # Williams %R (14-day), normalized to [0, 1]
    feat["williams_r"] = (
        (highest_14 - feat["close"]) / (highest_14 - lowest_14 + 1e-10)
    )

    # On-Balance Volume (normalized as ratio to its own moving average)
    obv = (np.sign(feat["close"].diff()) * feat["volume"]).fillna(0).cumsum()
    feat["obv_ratio"] = obv / (obv.rolling(21).mean().abs() + 1e-10)

    # Money Flow Index (14-day), normalized to [0, 1]
    typical_price = (feat["high"] + feat["low"] + feat["close"]) / 3
    money_flow = typical_price * feat["volume"]
    positive_flow = money_flow.where(typical_price.diff() > 0, 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price.diff() <= 0, 0).rolling(14).sum()
    feat["mfi"] = (
        100 - 100 / (1 + positive_flow / (negative_flow + 1e-10))
    ) / 100

    # Chaikin Money Flow (20-day)
    mf_multiplier = (
        ((feat["close"] - feat["low"]) - (feat["high"] - feat["close"]))
        / (feat["high"] - feat["low"] + 1e-10)
    )
    mf_volume = mf_multiplier * feat["volume"]
    feat["cmf"] = (
        mf_volume.rolling(20).sum() / (feat["volume"].rolling(20).sum() + 1e-10)
    )

    return feat


# Feature columns that we extract from the technical features DataFrame
FEATURE_COLUMNS = [
    "return", "log_return", "intraday_range", "gap", "volume_ratio",
    "sma_5_ratio", "sma_10_ratio", "sma_21_ratio",
    "ema_12_ratio", "ema_26_ratio",
    "macd", "bollinger_pos", "rsi",
    "volatility_5", "volatility_21", "atr",
    "close_position", "vwap_ratio", "txn_ratio",
    "roc_5", "roc_10", "stoch_k", "stoch_d",
    "williams_r", "obv_ratio", "mfi", "cmf",
]


def encode_sector(sector: str) -> float:
    """
    Encode a sector as a single normalized value in [0, 1].

    Looks up the sector's index in config.SECTORS and divides by
    (len(SECTORS) - 1) to produce a value from 0.0 to 1.0.

    Args:
        sector: Sector name (must be in config.SECTORS)

    Returns:
        Float in [0.0, 1.0]
    """
    sector_lower = sector.lower()
    if sector_lower in config.SECTORS:
        idx = config.SECTORS.index(sector_lower)
    else:
        idx = config.SECTORS.index("other")
        logger.warning(
            f"Unknown sector '{sector}', defaulting to 'other'. "
            f"Valid sectors: {config.SECTORS}"
        )
    num_sectors = len(config.SECTORS)
    if num_sectors <= 1:
        return 0.0
    return idx / (num_sectors - 1)


def build_feature_matrix(df: pd.DataFrame, sentiment_score: float = 0.0,
                          sector: str = "other",
                          lookback_days: int = None) -> np.ndarray:
    """
    Build the full feature matrix from OHLCV data.

    Takes the last `lookback_days` of data, computes technical features,
    and flattens into a single feature vector. Appends the normalized
    sector value and sentiment score.

    Args:
        df: DataFrame with OHLCV columns (and date)
        sentiment_score: Sentiment value from Anthropic API [-1, 1]
        sector: Sector name for encoding
        lookback_days: Number of days to include in the feature window
                       (default: TRAINING_WINDOW_DAYS from config)

    Returns:
        1D numpy array: the feature vector for the neural network
    """
    lookback_days = lookback_days or config.TRAINING_WINDOW_DAYS

    # Compute technical features
    feat_df = compute_technical_features(df)

    # Drop NaN rows from rolling calculations
    feat_df = feat_df.dropna(subset=FEATURE_COLUMNS)

    # Take the last lookback_days rows
    if len(feat_df) > lookback_days:
        feat_df = feat_df.tail(lookback_days)

    # Extract feature columns
    feature_matrix = feat_df[FEATURE_COLUMNS].values

    # Replace any remaining NaN/inf with 0
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

    # Flatten the 2D matrix (days x features) into a 1D vector
    flat_features = feature_matrix.flatten()

    # Append sector as a single normalized value [0, 1]
    sector_value = encode_sector(sector)
    flat_features = np.append(flat_features, sector_value)

    # Append sentiment score at the end
    flat_features = np.append(flat_features, sentiment_score)

    return flat_features.astype(np.float32)


def compute_targets(df: pd.DataFrame, current_idx: int) -> np.ndarray:
    """
    Compute the target values (actual future % price changes)
    for each prediction horizon.

    Args:
        df: Full DataFrame with 'close' column
        current_idx: Index position of the "current" day

    Returns:
        numpy array of shape (6,) with actual % changes (as fractions)
        for horizons: 1 day, 4 days, 1 week, 2 weeks, 1 month, 3 months.
        Returns None if not enough future data exists.
    """
    horizons = list(config.PREDICTION_HORIZONS.values())
    current_close = df.iloc[current_idx]["close"]
    targets = []

    for h in horizons:
        future_idx = current_idx + h
        if future_idx >= len(df):
            return None  # Not enough future data
        future_close = df.iloc[future_idx]["close"]
        pct_change = (future_close - current_close) / current_close
        # Clamp to [-1, 1] to match tanh output range
        pct_change = max(-1.0, min(1.0, pct_change))
        targets.append(pct_change)

    return np.array(targets, dtype=np.float32)


def get_feature_count(lookback_days: int = None) -> int:
    """
    Calculate the total number of input features.

    This equals (lookback_days * len(FEATURE_COLUMNS)) + 1 (sector) + 1 (sentiment).

    Args:
        lookback_days: Number of lookback days

    Returns:
        Total feature count (= input_size for the neural network)
    """
    lookback_days = lookback_days or config.TRAINING_WINDOW_DAYS
    return (lookback_days * len(FEATURE_COLUMNS)
            + 1   # sector (single normalized value)
            + 1)  # sentiment
