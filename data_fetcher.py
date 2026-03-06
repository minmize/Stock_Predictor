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

    def fetch_ticker_news(self, ticker: str, limit: int = 50) -> list[dict]:
        """
        Fetch recent news articles for a ticker via REST API.

        Args:
            ticker: Stock ticker symbol
            limit: Max number of articles (default: 50)

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

    def fetch_ticker_news_historical(self, ticker: str,
                                      before_date: str,
                                      limit: int = 10) -> list[dict]:
        """
        Fetch news articles for a ticker published before a given date.

        Used during training to get news that was available at a
        historical point in time (prevents data leakage).

        Args:
            ticker: Stock ticker symbol
            before_date: "YYYY-MM-DD" — only articles published on or before
            limit: Max number of articles (default: 10 for training)

        Returns:
            List of dicts with keys: title, description, published_utc, url
        """
        logger.info(
            f"REST: Fetching historical news for {ticker} before {before_date}"
        )
        articles = []
        try:
            for article in self.rest_client.list_ticker_news(
                ticker, limit=limit, sort="published_utc", order="desc",
                published_utc_lte=f"{before_date}T23:59:59Z"
            ):
                articles.append({
                    "title": article.title,
                    "description": getattr(article, "description", ""),
                    "published_utc": article.published_utc,
                    "article_url": article.article_url,
                })
        except Exception as e:
            logger.warning(
                f"Could not fetch historical news for {ticker}: {e}"
            )

        logger.info(
            f"REST: Got {len(articles)} historical news articles for {ticker}"
        )
        return articles

    def fetch_market_news_historical(self, before_date: str,
                                      limit: int = 10) -> list[dict]:
        """
        Fetch general market news published before a given date.

        Used during training for world events sentiment at a historical
        point in time.

        Args:
            before_date: "YYYY-MM-DD" — only articles published on or before
            limit: Max number of articles (default: 10 for training)

        Returns:
            List of dicts with keys: title, description, published_utc, url
        """
        logger.info(
            f"REST: Fetching historical market news before {before_date}"
        )
        articles = []
        for broad_ticker in ["SPY", "QQQ", "DIA"]:
            try:
                for article in self.rest_client.list_ticker_news(
                    broad_ticker, limit=limit // 3,
                    sort="published_utc", order="desc",
                    published_utc_lte=f"{before_date}T23:59:59Z"
                ):
                    articles.append({
                        "title": article.title,
                        "description": getattr(article, "description", ""),
                        "published_utc": article.published_utc,
                        "article_url": article.article_url,
                    })
            except Exception as e:
                logger.warning(
                    f"Could not fetch historical news for {broad_ticker}: {e}"
                )

        # Deduplicate by title
        seen_titles = set()
        unique_articles = []
        for article in articles:
            if article["title"] not in seen_titles:
                seen_titles.add(article["title"])
                unique_articles.append(article)

        logger.info(
            f"REST: Got {len(unique_articles)} historical market news articles"
        )
        return unique_articles[:limit]

    def fetch_market_news(self, limit: int = 50) -> list[dict]:
        """
        Fetch general market / world news (not ticker-specific).

        Uses broad market tickers (SPY, QQQ, DIA) to pull macro-level
        headlines that capture world events affecting markets.

        Args:
            limit: Max number of articles (default: 50)

        Returns:
            List of dicts with keys: title, description, published_utc, url
        """
        logger.info("REST: Fetching general market news")
        articles = []
        for broad_ticker in ["SPY", "QQQ", "DIA"]:
            try:
                for article in self.rest_client.list_ticker_news(
                    broad_ticker, limit=limit // 3,
                    sort="published_utc", order="desc"
                ):
                    articles.append({
                        "title": article.title,
                        "description": getattr(article, "description", ""),
                        "published_utc": article.published_utc,
                        "article_url": article.article_url,
                    })
            except Exception as e:
                logger.warning(f"Could not fetch news for {broad_ticker}: {e}")

        # Deduplicate by title
        seen_titles = set()
        unique_articles = []
        for article in articles:
            if article["title"] not in seen_titles:
                seen_titles.add(article["title"])
                unique_articles.append(article)

        logger.info(
            f"REST: Got {len(unique_articles)} general market news articles"
        )
        return unique_articles[:limit]

    def fetch_financials(self, ticker: str,
                         as_of_date: str = None) -> dict:
        """
        Fetch the most recent financial data for a ticker via
        Polygon's Stock Financials (vX) endpoint.

        For training: pass as_of_date to get financials that were
        actually available at that point in time (filed before that date).
        For prediction: omit as_of_date to get the latest filings.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")
            as_of_date: "YYYY-MM-DD" — only return filings filed on or
                        before this date (prevents data leakage in training)

        Returns:
            Dict of fundamental metrics. Returns empty dict on failure.
        """
        if as_of_date:
            logger.info(
                f"REST: Fetching financials for {ticker} as of {as_of_date}"
            )
        else:
            logger.info(f"REST: Fetching financials for {ticker} (latest)")

        try:
            kwargs = {
                "ticker": ticker,
                "limit": 2,
                "sort": "period_of_report_date",
                "order": "desc",
                "timeframe": "quarterly",
            }
            if as_of_date:
                kwargs["filing_date_lte"] = as_of_date

            financials_iter = self.rest_client.vx.list_stock_financials(
                **kwargs
            )
            financials_list = list(financials_iter)

            if not financials_list:
                logger.warning(f"No financial data for {ticker}")
                return {}

            latest = financials_list[0]
            prior = financials_list[1] if len(financials_list) > 1 else None

            result = {}

            # Extract from financials object
            fin = getattr(latest, "financials", {})
            income = fin.get("income_statement", {}) if isinstance(fin, dict) else {}
            balance = fin.get("balance_sheet", {}) if isinstance(fin, dict) else {}
            cash_flow = (
                fin.get("cash_flow_statement", {}) if isinstance(fin, dict) else {}
            )

            def _val(section, key):
                """Safely extract a numeric value from a financials section."""
                item = section.get(key, {})
                if isinstance(item, dict):
                    return item.get("value", 0.0)
                return 0.0

            revenue = _val(income, "revenues") or 1e-10
            net_income = _val(income, "net_income_loss")
            operating_income = _val(income, "operating_income_loss")
            eps_diluted = _val(income, "diluted_earnings_per_share")

            total_assets = _val(balance, "assets") or 1e-10
            total_liabilities = _val(balance, "liabilities")
            equity = _val(balance, "equity") or 1e-10
            current_assets = _val(balance, "current_assets") or 1e-10
            current_liabilities = (
                _val(balance, "current_liabilities") or 1e-10
            )

            operating_cf = _val(
                cash_flow, "net_cash_flow_from_operating_activities"
            )
            capex = abs(
                _val(cash_flow, "net_cash_flow_from_investing_activities")
            )
            dividends = abs(
                _val(cash_flow, "net_cash_flow_from_financing_activities")
            )

            # Compute metrics
            result["eps"] = float(eps_diluted)
            result["profit_margin"] = float(net_income / revenue)
            result["operating_margin"] = float(operating_income / revenue)
            result["debt_to_equity"] = float(total_liabilities / equity)
            result["current_ratio"] = float(current_assets / current_liabilities)
            result["roe"] = float(net_income / equity)
            result["asset_turnover"] = float(revenue / total_assets)
            result["free_cash_flow_margin"] = float(
                (operating_cf - capex) / revenue
            )
            result["payout_ratio"] = float(
                dividends / (abs(net_income) + 1e-10)
            )

            # Revenue growth (quarter over quarter)
            if prior:
                prior_fin = getattr(prior, "financials", {})
                prior_income = (
                    prior_fin.get("income_statement", {})
                    if isinstance(prior_fin, dict) else {}
                )
                prior_revenue = _val(prior_income, "revenues") or 1e-10
                result["revenue_growth"] = float(
                    (revenue - prior_revenue) / abs(prior_revenue)
                )
            else:
                result["revenue_growth"] = 0.0

            logger.info(
                f"Financials for {ticker}: "
                f"EPS={result['eps']:.2f}, "
                f"margin={result['profit_margin']:.2%}, "
                f"D/E={result['debt_to_equity']:.2f}"
            )
            return result

        except Exception as e:
            logger.warning(
                f"Could not fetch financials for {ticker}: {e}. "
                f"Using zeros."
            )
            return {}


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
