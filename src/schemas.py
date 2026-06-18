from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    message: str

class WordContribution(BaseModel):
    word: str
    weight: float
    tfidf: float
    contribution: float

class PredictResponse(BaseModel):
    prediction: str          # 'positive' or 'negative'
    probability: float       # probability of the predicted class
    word_contributions: List[WordContribution]
    unreliability_warnings: List[str]

class HealthResponse(BaseModel):
    status: str
    is_model_loaded: bool
