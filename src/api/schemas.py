"""Pydantic request / response schemas for the API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, ConfigDict


class HeartFeatures(BaseModel):
    """One patient's feature vector (UCI Heart Disease, 13 features)."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
        }
    })

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1=male, 0=female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=0, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")


class BatchRequest(BaseModel):
    instances: List[HeartFeatures]


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no_disease, 1 = disease")
    label: str
    confidence: float = Field(..., ge=0, le=1, description="Probability of predicted class")
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    api_version: str
