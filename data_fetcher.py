"""
Data fetcher module for Massive (formerly Polygon.io) REST API.

Fetches stock OHLCV data and news via the REST API for both
training and prediction modes.

REST API returns: open, high, low, close, volume, vwap, timestamp, transactions
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from polygon import RESTClient

import config

logger = logging.getLogger(__name__)


class MassiveDataFetcher:
    """Fetches stock data from Massive API via REST."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.MASSIVE_API_KEY
        self._rest_client = None

    @property
    def rest_client(self) -> RESTClient:
        """Lazy-initialize the REST client."""
        if self._rest_client is None:
            self._rest_client = RESTClient(self.api_key)
        return self._rest_client

    # ------------------------------------------------------------------
    # REST API methods
    # ------------------------------------------------------------------

    def fetch_rest_aggregates(self, ticker: str, from_date: str,
                              to_date: str, timespan: str = "day",
                              multiplier: int = 1) -> pd.DataFrame:
        """
        Fetch OHLCV aggregates via the REST API.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")
            from_date: Start date "YYYY-MM-DD"
            to_date: End date "YYYY-MM-DD"
            timespan: "day", "hour", "minute", etc.
            multiplier: Timespan multiplier

        Returns:
            DataFrame with columns: timestamp, open, high, low, close,
            volume, vwap, transactions
        """
        logger.info(f"REST: Fetching {ticker} aggregates {from_date} -> {to_date}")
        aggs = []
        for a in self.rest_client.list_aggs(
            ticker, multiplier, timespan, from_date, to_date, limit=50000
        ):
            aggs.append({
                "timestamp": a.timestamp,
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume,
                "vwap": a.vwap,
                "transactions": a.transactions,
            })

        if not aggs:
            logger.warning(f"No REST data returned for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame(aggs)
        # Convert millisecond timestamp to datetime
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("date").reset_index(drop=True)
        logger.info(f"REST: Got {len(df)} bars for {ticker}")
        return df

    def fetch_recent_data(self, ticker: str, months: int = 6) -> pd.DataFrame:
        """
        Fetch the most recent N months of daily data via REST API.
        Used for prediction mode. Defaults to 6 months to ensure enough
        data remains after dropping NaN rows from indicator warm-up
        (need at least TRAINING_WINDOW_DAYS clean rows).

        Args:
            ticker: Stock ticker symbol
            months: Number of months of history to fetch (default: 6)

        Returns:
            DataFrame of daily OHLCV data
        """
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=months * 31)).strftime("%Y-%m-%d")
        return self.fetch_rest_aggregates(ticker, from_date, to_date)

    def fetch_ticker_news(self, ticker: str, limit: int = 20) -> list[dict]:
        """
        Fetch recent news articles for a ticker via REST API.

        Args:
            ticker: Stock ticker symbol
            limit: Max number of articles

        Returns:
            List of dicts with keys: title, description, published_utc, url
        """
        logger.info(f"REST: Fetching news for {ticker}")
        articles = []
        for article in self.rest_client.list_ticker_news(
            ticker, limit=limit, sort="published_utc", order="desc"
        ):
            articles.append({
                "title": article.title,
                "description": getattr(article, "description", ""),
                "published_utc": article.published_utc,
                "article_url": article.article_url,
            })

        logger.info(f"REST: Got {len(articles)} news articles for {ticker}")
        return articles


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw data DataFrame into a standard format suitable
    for feature engineering.

    Handles REST format (timestamp, open, high, low, close, volume,
    vwap, transactions, date).

    Returns DataFrame with columns:
        date, open, high, low, close, volume, transactions, vwap (if available)
    """
    out = pd.DataFrame()

    if "date" in df.columns:
        out["date"] = pd.to_datetime(df["date"])
    elif "timestamp" in df.columns:
        out["date"] = pd.to_datetime(df["timestamp"], unit="ms")

    for col in ["open", "high", "low", "close", "volume", "transactions"]:
        if col in df.columns:
            out[col] = df[col].astype(float)

    if "vwap" in df.columns:
        out["vwap"] = df["vwap"].astype(float)

    out = out.sort_values("date").reset_index(drop=True)
    return out
