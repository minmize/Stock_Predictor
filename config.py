"""
Configuration for Stock Predictor.
Fill in your API keys before running.
"""

# ============================================================
# Massive (formerly Polygon.io) API Configuration
# ============================================================
# REST API key - obtain from https://massive.com/dashboard
MASSIVE_API_KEY = "YOUR_MASSIVE_API_KEY_HERE"

# S3 Flat Files credentials - obtain from your Massive dashboard
MASSIVE_S3_ACCESS_KEY = "YOUR_S3_ACCESS_KEY_HERE"
MASSIVE_S3_SECRET_KEY = "YOUR_S3_SECRET_KEY_HERE"
MASSIVE_S3_ENDPOINT = "https://files.massive.com"
MASSIVE_S3_BUCKET = "flatfiles"
MASSIVE_FLAT_FILE_PREFIX = "us_stocks_sip/day_aggs_v1"

# ============================================================
# Anthropic API Configuration
# ============================================================
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

# ============================================================
# Neural Network Defaults
# ============================================================
# Hidden layer sizes (adjustable)
HIDDEN_LAYERS = [200, 100, 30]

# Output timeframes in trading days
# 1 day, 4 days, 1 week (5), 2 weeks (10), 1 month (21), 3 months (63)
PREDICTION_HORIZONS = {
    "1_day": 1,
    "4_days": 4,
    "1_week": 5,
    "2_weeks": 10,
    "1_month": 21,
    "3_months": 63,
}

# Training parameters
LEARNING_RATE = 0.001
EPOCHS_PER_WINDOW = 50
BATCH_SIZE = 32
TRAINING_WINDOW_DAYS = 63  # 3 months of trading days

# Paths
WEIGHTS_DIR = "weights"
DATA_DIR = "data"
