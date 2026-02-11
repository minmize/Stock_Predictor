"""
Data fetcher module for Massive (formerly Polygon.io) API.

Supports two modes:
1. Flat file (S3) download - for bulk historical CSV data (training mode)
2. REST API - for recent data retrieval (prediction mode)

Flat file CSV columns: ticker, volume, open, close, high, low, window_start, transactions
REST API returns: open, high, low, close, volume, vwap, timestamp, transactions
"""

import os
import csv
import io
import gzip
import logging
from datetime import datetime, timedelta
from typing import Optional

import boto3
import pandas as pd
from botocore.config import Config
from polygon import RESTClient

import config

logger = logging.getLogger(__name__)


class MassiveDataFetcher:
    """Fetches stock data from Massive API via REST and S3 flat files."""

    def __init__(self, api_key: str = None, s3_access_key: str = None,
                 s3_secret_key: str = None):
        self.api_key = api_key or config.MASSIVE_API_KEY
        self.s3_access_key = s3_access_key or config.MASSIVE_S3_ACCESS_KEY
        self.s3_secret_key = s3_secret_key or config.MASSIVE_S3_SECRET_KEY
        self._rest_client = None
        self._s3_client = None

    @property
    def rest_client(self) -> RESTClient:
        """Lazy-initialize the REST client."""
        if self._rest_client is None:
            self._rest_client = RESTClient(self.api_key)
        return self._rest_client

    @property
    def s3_client(self):
        """Lazy-initialize the S3 client for flat file access."""
        if self._s3_client is None:
            session = boto3.Session(
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
            )
            self._s3_client = session.client(
                "s3",
                endpoint_url=config.MASSIVE_S3_ENDPOINT,
                config=Config(signature_version="s3v4"),
            )
        return self._s3_client

    # ------------------------------------------------------------------
    # REST API methods (for prediction mode - recent data)
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

    def fetch_recent_data(self, ticker: str, months: int = 3) -> pd.DataFrame:
        """
        Fetch the most recent N months of daily data via REST API.
        Used for prediction mode.

        Args:
            ticker: Stock ticker symbol
            months: Number of months of history to fetch

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

    # ------------------------------------------------------------------
    # Flat file (S3) methods (for training mode - bulk historical data)
    # ------------------------------------------------------------------

    def list_flat_files(self, year: int, month: int) -> list[str]:
        """
        List available flat file keys for a given year/month.

        Args:
            year: e.g. 2024
            month: e.g. 3

        Returns:
            List of S3 object keys
        """
        prefix = f"{config.MASSIVE_FLAT_FILE_PREFIX}/{year:04d}/{month:02d}/"
        logger.info(f"S3: Listing flat files with prefix {prefix}")
        keys = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=config.MASSIVE_S3_BUCKET, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        logger.info(f"S3: Found {len(keys)} files for {year:04d}/{month:02d}")
        return keys

    def download_flat_file(self, key: str) -> pd.DataFrame:
        """
        Download and parse a single flat file from S3.

        The flat files are gzip-compressed CSVs with columns:
        ticker, volume, open, close, high, low, window_start, transactions

        Args:
            key: S3 object key

        Returns:
            DataFrame of the flat file data
        """
        logger.info(f"S3: Downloading {key}")
        response = self.s3_client.get_object(
            Bucket=config.MASSIVE_S3_BUCKET, Key=key
        )
        body = response["Body"].read()

        # Flat files are gzip-compressed
        if key.endswith(".gz"):
            body = gzip.decompress(body)

        text = body.decode("utf-8")
        df = pd.read_csv(io.StringIO(text))
        logger.info(f"S3: Parsed {len(df)} rows from {key}")
        return df

    def download_ticker_flat_files(self, ticker: str, start_year: int,
                                   start_month: int, end_year: int,
                                   end_month: int) -> pd.DataFrame:
        """
        Download flat file data for a specific ticker across a date range.

        Iterates through month-by-month, downloads each flat file,
        filters for the requested ticker, and concatenates results.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")
            start_year: Starting year
            start_month: Starting month (1-12)
            end_year: Ending year
            end_month: Ending month (1-12)

        Returns:
            DataFrame filtered to the requested ticker, sorted by date
        """
        all_frames = []
        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)

        while current <= end:
            try:
                keys = self.list_flat_files(current.year, current.month)
                for key in keys:
                    df = self.download_flat_file(key)
                    if "ticker" in df.columns:
                        filtered = df[df["ticker"] == ticker].copy()
                        if not filtered.empty:
                            all_frames.append(filtered)
            except Exception as e:
                logger.warning(
                    f"S3: Error fetching {current.year}/{current.month}: {e}"
                )

            # Advance to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        if not all_frames:
            logger.warning(f"S3: No flat file data found for {ticker}")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)

        # Convert window_start (nanosecond epoch) to datetime
        if "window_start" in combined.columns:
            combined["date"] = pd.to_datetime(
                combined["window_start"], unit="ns"
            )
            combined = combined.sort_values("date").reset_index(drop=True)

        logger.info(
            f"S3: Total {len(combined)} rows for {ticker} "
            f"({start_year}/{start_month} -> {end_year}/{end_month})"
        )
        return combined

    def save_flat_data_to_csv(self, df: pd.DataFrame, ticker: str,
                              output_dir: str = None) -> str:
        """
        Save downloaded flat file data to a local CSV for caching.

        Args:
            df: DataFrame to save
            ticker: Ticker symbol (used in filename)
            output_dir: Directory to save in (default: config.DATA_DIR)

        Returns:
            Path to saved CSV file
        """
        output_dir = output_dir or config.DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{ticker}_flat_data.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved flat file data to {filepath}")
        return filepath

    def load_cached_csv(self, ticker: str,
                        data_dir: str = None) -> Optional[pd.DataFrame]:
        """
        Load previously cached flat file data from a local CSV.

        Args:
            ticker: Ticker symbol
            data_dir: Directory to look in (default: config.DATA_DIR)

        Returns:
            DataFrame if file exists, None otherwise
        """
        data_dir = data_dir or config.DATA_DIR
        filepath = os.path.join(data_dir, f"{ticker}_flat_data.csv")
        if os.path.exists(filepath):
            logger.info(f"Loading cached data from {filepath}")
            df = pd.read_csv(filepath)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        return None


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw data DataFrame into a standard format suitable
    for feature engineering.

    Handles both flat file format (ticker, volume, open, close, high, low,
    window_start, transactions) and REST format (timestamp, open, high, low,
    close, volume, vwap, transactions, date).

    Returns DataFrame with columns:
        date, open, high, low, close, volume, transactions, vwap (if available)
    """
    out = pd.DataFrame()

    if "date" in df.columns:
        out["date"] = pd.to_datetime(df["date"])
    elif "window_start" in df.columns:
        out["date"] = pd.to_datetime(df["window_start"], unit="ns")
    elif "timestamp" in df.columns:
        out["date"] = pd.to_datetime(df["timestamp"], unit="ms")

    for col in ["open", "high", "low", "close", "volume", "transactions"]:
        if col in df.columns:
            out[col] = df[col].astype(float)

    if "vwap" in df.columns:
        out["vwap"] = df["vwap"].astype(float)

    out = out.sort_values("date").reset_index(drop=True)
    return out
