import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import pickle
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import config
from schemas import PredictRequest, PredictResponse, WordContribution, HealthResponse

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_server")

app = FastAPI(
    title="Amazon Sentiment Classifier API",
    description="FastAPI service for real-time sentiment prediction, attribution, and unreliability checks.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and vectorizer
model = None
vectorizer = None

@app.on_event("startup")
def load_ml_components():
    """Load model and vectorizer at application startup."""
    global model, vectorizer
    logger.info("Initializing ML components...")
    
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.VECTORIZER_PATH):
        logger.error("Model or Vectorizer file not found. Please run src/train.py first.")
        return

    try:
        with open(config.MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(config.VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info("ML components successfully loaded into memory.")
    except Exception as e:
        logger.error(f"Error loading ML components: {e}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Verify application health and ML model status."""
    model_loaded = (model is not None) and (vectorizer is not None)
    status = "healthy" if model_loaded else "degraded"
    return HealthResponse(status=status, is_model_loaded=model_loaded)

@app.get("/api/metrics")
def get_metrics():
    """Retrieve pre-calculated training and evaluation metrics."""
    if not os.path.exists(config.METRICS_PATH):
        raise HTTPException(
            status_code=404, 
            detail="Metrics data not found. Please train the model first."
        )
    try:
        with open(config.METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Failed to read metrics file: {e}")
        raise HTTPException(status_code=500, detail="Internal error loading metrics.")

@app.get("/api/download-presentation")
def download_presentation():
    """Download the generated 2-page landscape project presentation PDF."""
    if not os.path.exists(config.PRESENTATION_PATH):
        raise HTTPException(
            status_code=404,
            detail="Presentation PDF not found. Please train the model first."
        )
    return FileResponse(
        path=config.PRESENTATION_PATH,
        filename="sentiment_presentation.pdf",
        media_type="application/pdf"
    )

@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Classify a product review and analyze feature attribution and model unreliability warnings."""
    global model, vectorizer
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model components are not loaded. Server is currently degraded."
        )

    try:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Empty review text.")

        # TF-IDF Transform
        vec = vectorizer.transform([message])
        vec_coo = vec.tocoo()
        tfidf_vals = {col: val for col, val in zip(vec_coo.col, vec_coo.data)}

        # Class Probability
        # LogisticRegression has classes ['negative', 'positive']
        # column 0 is negative, column 1 is positive
        prob_positive = float(model.predict_proba(vec)[0][1])
        prediction = "positive" if prob_positive >= 0.5 else "negative"
        confidence = prob_positive if prediction == "positive" else (1.0 - prob_positive)

        # Tokenize and compute word contributions
        raw_tokens = message.split()
        word_contributions = []
        oov_count = 0
        meaningful_words_count = 0

        # Basic negation list
        negation_words = {'not', 'no', 'never', 'neither', 'nor', 'dont', 'wasnt', 'cant', 'isnt', 'didnt', 'without', 'but'}
        # Basic contrast list
        contrast_words = {'but', 'however', 'although', 'yet', 'though'}

        has_negation = False
        has_contrast = False

        for token in raw_tokens:
            # Clean punctuation for vocabulary lookup
            cleaned_token = re.sub(r'[^\w]', '', token).lower()
            
            # Check for negation/contrast
            if cleaned_token in negation_words:
                has_negation = True
            if cleaned_token in contrast_words:
                has_contrast = True

            coef = 0.0
            tfidf_val = 0.0
            contribution = 0.0

            if cleaned_token:
                meaningful_words_count += 1
                if cleaned_token in vectorizer.vocabulary_:
                    vocab_idx = vectorizer.vocabulary_[cleaned_token]
                    coef = float(model.coef_[0][vocab_idx])
                    tfidf_val = float(tfidf_vals.get(vocab_idx, 0.0))
                    contribution = coef * tfidf_val
                else:
                    # OOV word (slang, Hinglish, typo)
                    # We only count it as OOV if it's not purely a number
                    if not cleaned_token.isdigit():
                        oov_count += 1

            word_contributions.append(
                WordContribution(
                    word=token,
                    weight=coef,
                    tfidf=tfidf_val,
                    contribution=contribution
                )
            )

        # Generate warnings
        unreliability_warnings = []
        
        # 1. Short Review Warning
        if meaningful_words_count < 4:
            unreliability_warnings.append("Vague review length: The review is extremely short (< 4 words). Model predictions on short reviews are often unreliable due to lack of contextual detail.")
            
        # 2. Negation Warning
        if has_negation:
            unreliability_warnings.append("Negation detected: The review contains negation words ('not', 'no', etc.). A bag-of-words/TF-IDF model cannot understand clause associations (e.g. misinterpreting 'not bad' or 'no issues') and may misclassify.")
            
        # 3. Contrast Warning (Mixed Sentiment)
        if has_contrast:
            unreliability_warnings.append("Contrastive clause detected: The review contains mixed sentiments (using 'but', 'however', etc.). The model combines word scores together and often misrepresents multi-clause opinions.")

        # 4. Hinglish / Typo / Out-of-Vocabulary Warning
        if meaningful_words_count >= 3:
            oov_rate = oov_count / meaningful_words_count
            if oov_rate > 0.35:
                unreliability_warnings.append(f"High Out-of-Vocabulary Rate ({oov_rate:.1%}): A large portion of words are unknown. This suggests Hinglish code-mixing ('acha tha', 'bakwas'), typos, or specific slang, which the model completely ignores.")

        logger.info(f"Prediction made: label={prediction}, confidence={confidence:.4f}, warnings={len(unreliability_warnings)}")
        
        return PredictResponse(
            prediction=prediction,
            probability=confidence,
            word_contributions=word_contributions,
            unreliability_warnings=unreliability_warnings
        )
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Inference failure.")

# Mount static folder to serve frontend dashboard
if os.path.exists(config.STATIC_DIR):
    app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
else:
    logger.warning(f"Static directory {config.STATIC_DIR} not found. UI files will not be served.")
