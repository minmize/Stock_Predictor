#!/usr/bin/env python3
"""
Stock Predictor - Neural Network Stock Price Prediction

Two modes of operation:
  1. train  - Download historical data via REST API, train in 3-month batches
  2. predict - Fetch recent data and output price predictions

Uses:
  - Massive (formerly Polygon.io) REST API for stock market data
  - Anthropic Claude API for news sentiment analysis
  - PyTorch neural network with universal weights (shared across all stocks)
  - Sector auto-detected via Ticker Details API (SIC code), normalized to [0,1]

Usage:
  python main.py train --ticker AAPL
  python main.py train --ticker AAPL --start 2020-01 --end 2024-12
  python main.py predict --ticker AAPL
  python main.py predict --ticker AAPL --sector technology --no-sentiment
  python main.py list
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import config
from data_fetcher import MassiveDataFetcher
from trainer import StockTrainer
from predictor import StockPredictor, format_prediction_report
from neural_net import ModelManager


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_api_keys(mode: str, use_sentiment: bool):
    """Check that required API keys are configured."""
    errors = []

    if config.MASSIVE_API_KEY == "YOUR_MASSIVE_API_KEY_HERE":
        errors.append(
            "MASSIVE_API_KEY not set in config.py "
            "(required for REST API data fetching)"
        )

    if use_sentiment:
        if config.ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY_HERE":
            errors.append(
                "ANTHROPIC_API_KEY not set in config.py "
                "(required for sentiment analysis, or use --no-sentiment)"
            )

    if errors:
        print("\nConfiguration errors:")
        for e in errors:
            print(f"  - {e}")
        print(
            "\nPlease edit config.py and fill in your API keys.\n"
            "See README or config.py comments for details."
        )
        sys.exit(1)


def resolve_sector(ticker: str, sector_override: str = None) -> str:
    """
    Resolve the sector for a ticker.

    If sector_override is provided, validate and return it.
    Otherwise auto-detect via the Ticker Details API.

    Args:
        ticker: Stock ticker symbol
        sector_override: Manual sector override (optional)

    Returns:
        Validated sector name from config.SECTORS
    """
    if sector_override:
        sector_lower = sector_override.lower()
        if sector_lower in config.SECTORS:
            return sector_lower
        print(f"\nWarning: Unknown sector '{sector_override}'.")
        print(f"Valid sectors: {', '.join(config.SECTORS)}")
        print(f"Defaulting to 'other'.\n")
        return "other"

    # Auto-detect via API
    print(f"  Looking up sector for {ticker} via API...")
    fetcher = MassiveDataFetcher()
    sector = fetcher.fetch_ticker_sector(ticker)
    print(f"  Detected sector: {sector}")
    return sector


def parse_year_month(date_str: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' into (year, month)."""
    parts = date_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM.")
    return int(parts[0]), int(parts[1])


def cmd_train(args):
    """Handle the 'train' subcommand."""
    use_sentiment = not args.no_sentiment
    validate_api_keys("train", use_sentiment)

    # Resolve dates: default to 4 years back from today
    if args.end:
        end_year, end_month = parse_year_month(args.end)
        end_date = datetime(end_year, end_month, 28)
    else:
        end_date = datetime.now()

    if args.start:
        start_year, start_month = parse_year_month(args.start)
        start_date = datetime(start_year, start_month, 1)
    else:
        start_date = end_date - timedelta(days=4 * 365)

    # Resolve sector
    sector = resolve_sector(args.ticker, args.sector)

    hidden_layers = None
    if args.hidden_layers:
        hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    print(f"\n{'='*60}")
    print(f"  TRAINING MODE - {args.ticker}")
    print(f"{'='*60}")
    print(f"  Date range:    {start_date.date()} to {end_date.date()}")
    print(f"  Sector:        {sector}")
    print(f"  Model:         Universal (shared across all stocks)")
    print(f"  Training:      3-month batches")
    print(f"  Sentiment:     {'Enabled' if use_sentiment else 'Disabled'}")
    print(f"  Hidden layers: {hidden_layers or config.HIDDEN_LAYERS}")
    print(f"  Epochs/batch:  {args.epochs}")
    print(f"{'='*60}\n")

    trainer = StockTrainer(
        args.ticker, sector=sector, hidden_layers=hidden_layers,
        use_sentiment=use_sentiment,
    )
    results = trainer.run_training(
        end_date=end_date,
        start_date=start_date,
        epochs_per_batch=args.epochs,
    )

    if results:
        print(f"\nTraining complete! {len(results)} batches processed.")
        final = results[-1]
        print(f"Final validation loss: {final['final_val_loss']:.6f}")
        print(f"Per-horizon MAE:")
        for name, mae in final["per_horizon_mae"].items():
            print(f"  {name}: {mae:.4f} ({mae*100:.2f}%)")
    else:
        print("\nTraining failed. Check logs for details.")


def cmd_predict(args):
    """Handle the 'predict' subcommand."""
    use_sentiment = not args.no_sentiment
    validate_api_keys("predict", use_sentiment)

    # Resolve sector (auto-detect or override)
    sector = resolve_sector(args.ticker, args.sector)

    hidden_layers = None
    if args.hidden_layers:
        hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    print(f"\nGenerating predictions for {args.ticker} (sector: {sector})...")

    predictor = StockPredictor(
        args.ticker, sector=sector, hidden_layers=hidden_layers,
        use_sentiment=use_sentiment,
    )
    result = predictor.predict()

    if result:
        report = format_prediction_report(result)
        print(report)
    else:
        print(
            f"\nPrediction failed for {args.ticker}. "
            f"Ensure the universal model has been trained first."
        )

    # Check for saved model
    manager = ModelManager()
    if not manager.has_saved_model():
        print("\nNo universal model found.")
        print(f"Run: python main.py train --ticker {args.ticker}")


def cmd_list(args):
    """Handle the 'list' subcommand - show saved universal model info."""
    manager = ModelManager()
    if manager.has_saved_model():
        model = manager.load_model()
        if model:
            info = model.get_architecture_info()
            print("\nUniversal model:")
            print(
                f"  Parameters: {info['total_params']}, "
                f"input={info['input_size']}, "
                f"hidden={info['hidden_layers']}, "
                f"output={info['num_outputs']}"
            )
            meta = manager.get_model_info()
            if meta:
                training_info = meta.get("training_info", {})
                if training_info:
                    print(f"  Last trained on: {training_info.get('ticker', '?')} "
                          f"(sector: {training_info.get('sector', '?')})")
                    print(f"  Samples: {training_info.get('num_samples', '?')}")
                    print(f"  Best val loss: "
                          f"{training_info.get('best_val_loss', '?')}")
        else:
            print("\nUniversal model found but could not be loaded.")
    else:
        print("\nNo universal model found. Run training first.")

    print(f"\nAvailable sectors: {', '.join(config.SECTORS)}")


def main():
    parser = argparse.ArgumentParser(
        description="Stock Predictor - Neural Network Price Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults (4 years back, sector auto-detected)
  python main.py train --ticker AAPL

  # Train with specific date range
  python main.py train --ticker AAPL --start 2020-01 --end 2024-12

  # Train with sector override
  python main.py train --ticker AAPL --sector technology

  # Train on another stock (same universal model continues learning)
  python main.py train --ticker JPM

  # Train without sentiment analysis
  python main.py train --ticker AAPL --no-sentiment

  # Predict (sector auto-detected, universal model)
  python main.py predict --ticker AAPL

  # Predict with sector override
  python main.py predict --ticker AAPL --sector technology

  # Custom hidden layers
  python main.py train --ticker AAPL --hidden-layers 256,128,64,16

  # List saved model info
  python main.py list

Available sectors:
  technology, healthcare, financial, consumer_discretionary,
  consumer_staples, energy, industrials, materials, utilities,
  real_estate, communication, other
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Mode of operation")

    # --- Train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train the neural network")
    train_parser.add_argument(
        "--ticker", "-t", required=True, help="Stock ticker symbol (e.g. AAPL)"
    )
    train_parser.add_argument(
        "--sector",
        help="Sector override (auto-detected via API if omitted). "
             "Options: technology, healthcare, financial, etc.",
    )
    train_parser.add_argument(
        "--start", "-s",
        help="Start date in YYYY-MM format (default: 4 years before end date)",
    )
    train_parser.add_argument(
        "--end", "-e",
        help="End date in YYYY-MM format (default: current date)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS_PER_WINDOW,
        help=f"Epochs per 3-month batch (default: {config.EPOCHS_PER_WINDOW})",
    )
    train_parser.add_argument(
        "--no-sentiment", action="store_true",
        help="Disable sentiment analysis (no Anthropic API needed)",
    )
    train_parser.add_argument(
        "--hidden-layers",
        help="Comma-separated hidden layer sizes (e.g. 200,100,30)",
    )
    train_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging",
    )
    train_parser.set_defaults(func=cmd_train)

    # --- Predict subcommand ---
    predict_parser = subparsers.add_parser(
        "predict", help="Predict price movements"
    )
    predict_parser.add_argument(
        "--ticker", "-t", required=True, help="Stock ticker symbol (e.g. AAPL)"
    )
    predict_parser.add_argument(
        "--sector",
        help="Sector override (auto-detected via API if omitted)",
    )
    predict_parser.add_argument(
        "--no-sentiment", action="store_true",
        help="Disable sentiment analysis",
    )
    predict_parser.add_argument(
        "--hidden-layers",
        help="Comma-separated hidden layer sizes (must match training)",
    )
    predict_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging",
    )
    predict_parser.set_defaults(func=cmd_predict)

    # --- List subcommand ---
    list_parser = subparsers.add_parser("list", help="List saved model info")
    list_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
