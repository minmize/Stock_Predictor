"""
Data fetcher module for Massive (formerly Polygon.io) REST API.

Fetches stock OHLCV data, ticker details (including sector via SIC code),
and news via the REST API for both training and prediction modes.

REST API returns: open, high, low, close, volume, vwap, timestamp, transactions
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from polygon import RESTClient

import config

logger = logging.getLogger(__name__)


# SIC code ranges mapped to our sector categories.
# Uses the first 2 digits (SIC "Major Group") for broad classification,
# with finer 4-digit overrides for ambiguous manufacturing ranges.
_SIC_SECTOR_MAP = [
    # Agriculture, Forestry, Fishing (01-09)
    (100, 999, "other"),
    # Mining (10-14)
    (1000, 1499, "energy"),  # most mining tickers are energy-adjacent
    # Construction (15-17)
    (1500, 1799, "industrials"),
    # Manufacturing: Food & Tobacco (20-21)
    (2000, 2199, "consumer_staples"),
    # Manufacturing: Textiles, Apparel, Lumber, Furniture (22-25)
    (2200, 2599, "consumer_discretionary"),
    # Manufacturing: Paper, Printing (26-27)
    (2600, 2799, "materials"),
    # Manufacturing: Chemicals & Pharmaceuticals (28)
    (2800, 2829, "materials"),       # industrial chemicals
    (2830, 2869, "healthcare"),      # drugs & pharma
    (2870, 2899, "materials"),       # agricultural chemicals, misc
    # Manufacturing: Petroleum Refining (29)
    (2900, 2999, "energy"),
    # Manufacturing: Rubber, Plastics, Stone, Metals (30-34)
    (3000, 3499, "materials"),
    # Manufacturing: Machinery & Computers (35)
    (3500, 3599, "technology"),
    # Manufacturing: Electronics & Electrical (36)
    (3600, 3699, "technology"),
    # Manufacturing: Transportation Equipment (37)
    (3700, 3799, "industrials"),
    # Manufacturing: Instruments (38)
    (3800, 3839, "technology"),
    (3840, 3859, "healthcare"),      # medical instruments
    (3860, 3899, "technology"),
    # Manufacturing: Misc (39)
    (3900, 3999, "industrials"),
    # Transportation (40-47)
    (4000, 4799, "industrials"),
    # Communications (48)
    (4800, 4899, "communication"),
    # Utilities (49)
    (4900, 4999, "utilities"),
    # Wholesale Trade (50-51)
    (5000, 5199, "industrials"),
    # Retail Trade (52-59)
    (5200, 5999, "consumer_discretionary"),
    # Finance, Insurance (60-64)
    (6000, 6499, "financial"),
    # Real Estate (65)
    (6500, 6599, "real_estate"),
    # Holding & Investment Offices (67)
    (6700, 6799, "financial"),
    # Services: Hotels, Personal, Business, Auto (70-76)
    (7000, 7299, "consumer_discretionary"),
    (7300, 7399, "technology"),      # computer & data services
    (7400, 7699, "consumer_discretionary"),
    # Services: Amusement, Health, Legal, Education, Social, Engineering (78-89)
    (7800, 7999, "communication"),   # amusement, recreation, media
    (8000, 8099, "healthcare"),      # health services
    (8100, 8999, "consumer_discretionary"),
    # Public Administration (91-99)
    (9100, 9999, "other"),
]


def sic_to_sector(sic_code: int) -> str:
    """
    Map a 4-digit SIC code to one of our sector categories.

    Args:
        sic_code: Standard Industrial Classification code (4-digit int)

    Returns:
        Sector name from config.SECTORS
    """
    for low, high, sector in _SIC_SECTOR_MAP:
        if low <= sic_code <= high:
            return sector
    return "other"


def sic_description_to_sector(description: str) -> str:
    """
    Fallback: map a SIC description string to a sector using keywords.

    Args:
        description: SIC description text (e.g. "ELECTRONIC COMPUTERS")

    Returns:
        Sector name from config.SECTORS
    """
    desc = description.lower()

    keyword_map = {
        "technology": [
            "computer", "software", "semiconductor", "electronic",
            "data processing", "programming", "circuit", "telecom",
        ],
        "healthcare": [
            "pharma", "drug", "medical", "health", "biotech",
            "surgical", "dental", "hospital",
        ],
        "financial": [
            "bank", "insurance", "credit", "loan", "securi",
            "invest", "financ", "mortgage",
        ],
        "energy": [
            "petroleum", "crude oil", "natural gas", "coal",
            "oil", "energy", "refin",
        ],
        "consumer_staples": [
            "food", "beverage", "tobacco", "grocery", "soap",
            "household",
        ],
        "consumer_discretionary": [
            "retail", "restaurant", "apparel", "auto", "hotel",
            "clothing", "department store",
        ],
        "communication": [
            "broadcast", "cable", "television", "radio", "media",
            "motion picture", "publish", "newspaper",
        ],
        "utilities": ["electric", "gas distribut", "water supply", "utilit"],
        "real_estate": ["real estate", "property", "reit"],
        "industrials": [
            "aerospace", "defense", "freight", "railroad", "trucking",
            "machinery", "construction", "engineering",
        ],
        "materials": [
            "chemical", "steel", "metal", "mining", "paper",
            "plastic", "glass", "cement", "lumber",
        ],
    }
    for sector, keywords in keyword_map.items():
        if any(kw in desc for kw in keywords):
            return sector
    return "other"


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
    # Ticker details & sector lookup
    # ------------------------------------------------------------------

    def fetch_ticker_sector(self, ticker: str) -> str:
        """
        Look up the sector for a ticker via the Ticker Details API.

        Uses the SIC code and SIC description returned by Polygon's
        Ticker Details v3 endpoint to map to one of our sector categories.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")

        Returns:
            Sector name from config.SECTORS (defaults to "other")
        """
        try:
            details = self.rest_client.get_ticker_details(ticker)
            sic_code = getattr(details, "sic_code", None)
            sic_desc = getattr(details, "sic_description", None)

            if sic_code:
                sector = sic_to_sector(int(sic_code))
                logger.info(
                    f"Sector for {ticker}: {sector} "
                    f"(SIC {sic_code}: {sic_desc or 'N/A'})"
                )
                return sector

            if sic_desc:
                sector = sic_description_to_sector(sic_desc)
                logger.info(
                    f"Sector for {ticker}: {sector} "
                    f"(from description: {sic_desc})"
                )
                return sector

            logger.warning(
                f"No SIC code or description for {ticker}, "
                f"defaulting to 'other'"
            )
            return "other"

        except Exception as e:
            logger.warning(
                f"Could not fetch ticker details for {ticker}: {e}. "
                f"Defaulting to 'other'"
            )
            return "other"

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
