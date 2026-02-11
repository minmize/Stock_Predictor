#!/usr/bin/env python3
"""
Stock Predictor - Neural Network Stock Price Prediction

Two modes of operation:
  1. train  - Download historical data, train the neural network
  2. predict - Fetch recent data and output price predictions

Uses:
  - Massive (formerly Polygon.io) API for stock market data
  - Anthropic Claude API for news sentiment analysis
  - PyTorch neural network with dynamic input sizing

Usage:
  python main.py train --ticker AAPL --start 2020-01 --end 2024-12
  python main.py predict --ticker AAPL
  python main.py train --ticker AAPL --start 2020-01 --end 2024-12 --use-rest
  python main.py predict --ticker AAPL --no-sentiment
"""

import argparse
import logging
import sys

import config
from trainer import StockTrainer, run_incremental_training
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


def validate_api_keys(mode: str, use_sentiment: bool, use_rest: bool = False):
    """Check that required API keys are configured."""
    errors = []

    if config.MASSIVE_API_KEY == "YOUR_MASSIVE_API_KEY_HERE":
        if mode == "predict":
            errors.append(
                "MASSIVE_API_KEY not set in config.py "
                "(required for REST API data fetching)"
            )
        elif mode == "train":
            errors.append(
                "MASSIVE_API_KEY not set in config.py "
                "(required for data fetching)"
            )

    if mode == "train":
        if (config.MASSIVE_S3_ACCESS_KEY == "YOUR_S3_ACCESS_KEY_HERE"
                and not use_rest):
            logging.warning(
                "S3 keys not configured. Use --use-rest flag to fetch "
                "training data via REST API instead of flat files."
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


def parse_year_month(date_str: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' into (year, month)."""
    parts = date_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM.")
    return int(parts[0]), int(parts[1])


def cmd_train(args):
    """Handle the 'train' subcommand."""
    start_year, start_month = parse_year_month(args.start)
    end_year, end_month = parse_year_month(args.end)
    use_sentiment = not args.no_sentiment

    validate_api_keys("train", use_sentiment, use_rest=args.use_rest)

    hidden_layers = None
    if args.hidden_layers:
        hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    print(f"\n{'='*60}")
    print(f"  TRAINING MODE - {args.ticker}")
    print(f"{'='*60}")
    print(f"  Data range:  {args.start} to {args.end}")
    print(f"  Data source: {'REST API' if args.use_rest else 'S3 Flat Files'}")
    print(f"  Sentiment:   {'Enabled' if use_sentiment else 'Disabled'}")
    print(f"  Hidden layers: {hidden_layers or config.HIDDEN_LAYERS}")
    print(f"  Epochs/window: {args.epochs}")
    print(f"{'='*60}\n")

    if args.incremental:
        results = run_incremental_training(
            ticker=args.ticker,
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            hidden_layers=hidden_layers,
            use_rest=args.use_rest,
            use_sentiment=use_sentiment,
            epochs_per_window=args.epochs,
        )
        if results:
            print(f"\nTraining complete! {len(results)} windows processed.")
            final = results[-1]
            print(f"Final validation loss: {final['final_val_loss']:.6f}")
        else:
            print("\nTraining failed. Check logs for details.")
    else:
        trainer = StockTrainer(
            args.ticker, hidden_layers=hidden_layers,
            use_sentiment=use_sentiment,
        )
        result = trainer.run_full_training(
            start_year, start_month, end_year, end_month,
            use_rest=args.use_rest, epochs=args.epochs,
        )
        if result:
            print(f"\nTraining complete!")
            print(f"  Final train loss: {result['final_train_loss']:.6f}")
            print(f"  Final val loss:   {result['final_val_loss']:.6f}")
            print(f"  Per-horizon MAE:")
            for name, mae in result["per_horizon_mae"].items():
                print(f"    {name}: {mae:.4f} ({mae*100:.2f}%)")
        else:
            print("\nTraining failed. Check logs for details.")


def cmd_predict(args):
    """Handle the 'predict' subcommand."""
    use_sentiment = not args.no_sentiment

    validate_api_keys("predict", use_sentiment)

    hidden_layers = None
    if args.hidden_layers:
        hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    print(f"\nGenerating predictions for {args.ticker}...")

    predictor = StockPredictor(
        args.ticker, hidden_layers=hidden_layers,
        use_sentiment=use_sentiment,
    )
    result = predictor.predict()

    if result:
        report = format_prediction_report(result)
        print(report)
    else:
        print(
            f"\nPrediction failed for {args.ticker}. "
            f"Ensure the model has been trained first."
        )

    # Check for saved models
    manager = ModelManager()
    saved = manager.list_saved_models()
    if saved and args.ticker not in saved:
        print(f"\nAvailable trained models: {', '.join(saved)}")
        print(f"Run: python main.py train --ticker {args.ticker} ... first")


def cmd_list(args):
    """Handle the 'list' subcommand - show saved models."""
    manager = ModelManager()
    saved = manager.list_saved_models()
    if saved:
        print("\nSaved models:")
        for ticker in saved:
            model = manager.load_model(ticker)
            if model:
                info = model.get_architecture_info()
                print(
                    f"  {ticker}: {info['total_params']} params, "
                    f"input={info['input_size']}, "
                    f"hidden={info['hidden_layers']}"
                )
            else:
                print(f"  {ticker}: (metadata only)")
    else:
        print("\nNo saved models found. Run training first.")


def main():
    parser = argparse.ArgumentParser(
        description="Stock Predictor - Neural Network Price Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on AAPL using flat files from 2020 to 2024
  python main.py train --ticker AAPL --start 2020-01 --end 2024-12

  # Train using REST API (no S3 keys needed)
  python main.py train --ticker AAPL --start 2023-01 --end 2024-12 --use-rest

  # Train incrementally in 3-month windows
  python main.py train --ticker AAPL --start 2020-01 --end 2024-12 --incremental

  # Train without sentiment analysis
  python main.py train --ticker AAPL --start 2023-01 --end 2024-12 --no-sentiment --use-rest

  # Predict current price movements
  python main.py predict --ticker AAPL

  # Predict without sentiment
  python main.py predict --ticker AAPL --no-sentiment

  # Custom hidden layers
  python main.py train --ticker AAPL --start 2023-01 --end 2024-12 --hidden-layers 256,128,64,16

  # List saved models
  python main.py list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Mode of operation")

    # --- Train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train the neural network")
    train_parser.add_argument(
        "--ticker", "-t", required=True, help="Stock ticker symbol (e.g. AAPL)"
    )
    train_parser.add_argument(
        "--start", "-s", required=True,
        help="Start date in YYYY-MM format (e.g. 2020-01)",
    )
    train_parser.add_argument(
        "--end", "-e", required=True,
        help="End date in YYYY-MM format (e.g. 2024-12)",
    )
    train_parser.add_argument(
        "--use-rest", action="store_true",
        help="Use REST API instead of S3 flat files for data fetching",
    )
    train_parser.add_argument(
        "--incremental", action="store_true",
        help="Train incrementally in 3-month sliding windows",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS_PER_WINDOW,
        help=f"Epochs per training window (default: {config.EPOCHS_PER_WINDOW})",
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
    list_parser = subparsers.add_parser("list", help="List saved models")
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
