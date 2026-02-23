"""
Sentiment evaluator using Anthropic's Claude API.

Analyzes recent news articles for a stock's sector and produces a
sentiment score from -1.0 (extremely negative) to +1.0 (extremely positive).
This score is used as an input feature for the neural network.
"""

import logging
from typing import Optional

import anthropic

import config

logger = logging.getLogger(__name__)

# Sector mapping for common tickers - extend as needed
TICKER_SECTORS = {
    "AAPL": "Technology / Consumer Electronics",
    "MSFT": "Technology / Software",
    "GOOGL": "Technology / Internet / Advertising",
    "GOOG": "Technology / Internet / Advertising",
    "AMZN": "Technology / E-Commerce / Cloud Computing",
    "META": "Technology / Social Media",
    "TSLA": "Automotive / Electric Vehicles / Energy",
    "NVDA": "Technology / Semiconductors / AI",
    "JPM": "Financial Services / Banking",
    "JNJ": "Healthcare / Pharmaceuticals",
    "V": "Financial Services / Payments",
    "WMT": "Retail / Consumer Staples",
    "XOM": "Energy / Oil & Gas",
    "UNH": "Healthcare / Insurance",
    "BAC": "Financial Services / Banking",
    "PG": "Consumer Staples / Household Products",
    "HD": "Retail / Home Improvement",
    "DIS": "Entertainment / Media",
    "NFLX": "Entertainment / Streaming",
    "AMD": "Technology / Semiconductors",
    "INTC": "Technology / Semiconductors",
    "CRM": "Technology / Enterprise Software",
}

SENTIMENT_SYSTEM_PROMPT = """You are a financial sentiment analyst. You will be given
a set of recent news article titles and descriptions related to a specific stock
and its industry sector.

Your task is to analyze the overall sentiment and return ONLY a single floating
point number between -1.0 and 1.0 where:

-1.0 = Extremely negative sentiment (catastrophic news, major scandals, bankruptcy)
-0.5 = Moderately negative (earnings miss, lawsuits, downgrades)
-0.2 = Slightly negative (minor concerns, slight headwinds)
 0.0 = Neutral (mixed or no significant news)
 0.2 = Slightly positive (minor positive developments)
 0.5 = Moderately positive (earnings beat, upgrades, new products)
 1.0 = Extremely positive (breakthrough developments, massive growth)

Consider:
- The overall tone of headlines and descriptions
- Potential market impact of the news
- Whether news is about the specific company vs. broader sector trends
- Recency and relevance of the articles

Return ONLY the number, nothing else. Example: 0.35"""


class SentimentEvaluator:
    """Evaluates market sentiment using Anthropic Claude API."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.model = model or config.ANTHROPIC_MODEL
        self._client = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def get_sector(self, ticker: str) -> str:
        """
        Get the sector/industry for a ticker.

        Falls back to asking Claude if the ticker isn't in our mapping.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Sector description string
        """
        if ticker in TICKER_SECTORS:
            return TICKER_SECTORS[ticker]

        # Ask Claude for the sector if we don't have it cached
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": (
                        f"What industry sector does the stock ticker {ticker} "
                        f"belong to? Reply with ONLY the sector name, e.g. "
                        f"'Technology / Semiconductors'. Keep it brief."
                    ),
                }],
            )
            sector = message.content[0].text.strip()
            TICKER_SECTORS[ticker] = sector  # Cache it
            return sector
        except Exception as e:
            logger.warning(f"Could not determine sector for {ticker}: {e}")
            return "Unknown Sector"

    def evaluate_sentiment(self, ticker: str,
                           news_articles: list[dict]) -> float:
        """
        Analyze news articles and return a sentiment score.

        Args:
            ticker: Stock ticker symbol
            news_articles: List of dicts with 'title' and 'description' keys

        Returns:
            Float between -1.0 and 1.0 representing sentiment
        """
        if not news_articles:
            logger.info(f"No news articles for {ticker}, returning neutral 0.0")
            return 0.0

        # Limit to most recent 50 articles to stay within API token limits
        if len(news_articles) > 50:
            logger.info(
                f"Trimming {len(news_articles)} articles to 50 for {ticker}"
            )
            news_articles = news_articles[:50]

        sector = self.get_sector(ticker)

        # Build the news summary for Claude
        news_text = f"Stock: {ticker}\nSector: {sector}\n\nRecent News:\n\n"
        for i, article in enumerate(news_articles, 1):
            title = article.get("title", "No title")
            desc = article.get("description", "No description")
            date = article.get("published_utc", "Unknown date")
            news_text += f"{i}. [{date}] {title}\n   {desc}\n\n"

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=20,
                system=SENTIMENT_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": news_text,
                }],
            )
            raw_score = message.content[0].text.strip()
            score = float(raw_score)
            # Clamp to [-1.0, 1.0]
            score = max(-1.0, min(1.0, score))
            logger.info(f"Sentiment for {ticker}: {score}")
            return score

        except (ValueError, IndexError) as e:
            logger.warning(
                f"Could not parse sentiment score for {ticker}: {e}. "
                f"Returning neutral."
            )
            return 0.0
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error for {ticker}: {e}")
            return 0.0

    def evaluate_sentiment_from_text(self, ticker: str,
                                     text_summaries: list[str]) -> float:
        """
        Convenience method: evaluate sentiment from plain text summaries
        instead of structured article dicts.

        Args:
            ticker: Stock ticker symbol
            text_summaries: List of news headline/summary strings

        Returns:
            Float between -1.0 and 1.0
        """
        articles = [{"title": s, "description": ""} for s in text_summaries]
        return self.evaluate_sentiment(ticker, articles)


def get_training_sentiment(ticker: str, date_str: str,
                           fetcher=None) -> float:
    """
    Get a sentiment score for a historical date during training.

    During training we may not have real historical news, so this function
    attempts to fetch news around the given date. If unavailable,
    returns 0.0 (neutral).

    Args:
        ticker: Stock ticker symbol
        date_str: Date string "YYYY-MM-DD"
        fetcher: Optional MassiveDataFetcher instance for news retrieval

    Returns:
        Sentiment score float
    """
    if fetcher is None:
        return 0.0

    try:
        articles = fetcher.fetch_ticker_news(ticker, limit=10)
        evaluator = SentimentEvaluator()
        return evaluator.evaluate_sentiment(ticker, articles)
    except Exception as e:
        logger.warning(f"Could not get training sentiment for {ticker}: {e}")
        return 0.0
