import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data & Model Paths
DATA_PATH = os.path.join(BASE_DIR, "archive (1)", "7817_1.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
PRESENTATION_PATH = os.path.join(MODEL_DIR, "sentiment_presentation.pdf")

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Static files directory
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Server Config
LOG_LEVEL = "INFO"
CORS_ORIGINS = ["*"]
