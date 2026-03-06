"""
Configuration for Stock Predictor.
Fill in your API keys before running.
"""

# ============================================================
# Massive (formerly Polygon.io) API Configuration
# ============================================================
# REST API key - obtain from https://massive.com/dashboard
MASSIVE_API_KEY = "Sh6gU3i3rGPRqm9VLot4zERlzrphM35o"

# ============================================================
# Anthropic API Configuration
# ============================================================
ANTHROPIC_API_KEY = "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"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

# ============================================================
# Sector Definitions (normalized index used as neural net input)
# ============================================================
SECTORS = [
    "technology",
    "healthcare",
    "financial",
    "consumer_discretionary",
    "consumer_staples",
    "energy",
    "industrials",
    "materials",
    "utilities",
    "real_estate",
    "communication",
    "other",
]

# ============================================================
# Neural Network Defaults
# ============================================================
# Hidden layer sizes (adjustable)
# Sized for ~1714 input features: gradual compression from input to output.
# Layer 1 (512): compresses 1714 inputs ~3.3:1, captures broad patterns
# Layer 2 (256): intermediate representation
# Layer 3 (128): higher-level abstractions
# Layer 4 (32):  compact representation before 6 output nodes
HIDDEN_LAYERS = [512, 256, 128, 32]

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

# Universal model name (same weights used for all stocks)
UNIVERSAL_MODEL_NAME = "universal"

# Paths
WEIGHTS_DIR = "weights"
DATA_DIR = "data"
