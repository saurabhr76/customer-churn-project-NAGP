"""
app.py — Customer Churn Prediction REST API
===========================================

A FastAPI service that exposes the trained Decision Tree pipeline.

    uvicorn app:app --reload --port 8000

Endpoints
---------
POST /predict         score a single customer   (required by the assignment)
POST /predict/batch   score up to 1000 customers in one call
GET  /health          liveness + model-loaded probe
GET  /model-info      model version, params, hold-out metrics, expected schema
GET  /docs            interactive Swagger UI (auto-generated)

Design notes
------------
* The saved artifact is a *bundle* (pipeline + threshold + metadata), so the API
  never re-implements a transformation: cleaning, the 8 engineered features,
  imputation and one-hot encoding all happen inside `pipeline.predict_proba`.
  That is what eliminates training/serving skew.
* The model is loaded ONCE at startup via the lifespan handler, not per request.
* Validation is declarative (Pydantic + Enums), so malformed input is rejected
  with a precise HTTP 422 before it ever reaches the model.
* The Yes/No label comes from the *stored business threshold*, not from
  `pipeline.predict()`, so the operating point chosen in the notebook's cost
  analysis is what the service actually applies.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# `churn_features` must be importable: joblib stored a *reference* to
# get_feature_frame, not its code. Importing it here also guarantees the module
# is registered before joblib.load() resolves the pickle.
import churn_features as cf  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("churn-api")

MODEL_PATH = os.getenv(
    "CHURN_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "churn_model.pkl"),
)

#: Populated at startup by the lifespan handler.
ARTIFACT: Dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Startup / shutdown
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the process starts."""
    try:
        bundle = joblib.load(MODEL_PATH)
        ARTIFACT.update(bundle)
        logger.info("Model loaded from %s", MODEL_PATH)
        logger.info("  version=%s  trained_at=%s  threshold=%.2f",
                    bundle.get("model_version"), bundle.get("trained_at"), bundle.get("threshold", 0.5))
    except FileNotFoundError:
        logger.error("Model file not found at %s — run `python train_model.py` first.", MODEL_PATH)
    except Exception:  # pragma: no cover
        logger.exception("Failed to load the model artifact.")
    yield
    ARTIFACT.clear()
    logger.info("Shutting down.")


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Predicts whether a telecom customer is likely to churn, and returns the "
        "churn probability plus a risk band the retention team can prioritise on."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Request schema — declarative validation
# --------------------------------------------------------------------------- #
class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class Gender(str, Enum):
    male = "Male"
    female = "Female"


class MultipleLinesOption(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone = "No phone service"


class InternetServiceType(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    no = "No"


class InternetAddOn(str, Enum):
    """Used by the six optional services, which share a 3-level vocabulary."""
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"


class ContractType(str, Enum):
    month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodType(str, Enum):
    e_check = "Electronic check"
    mailed = "Mailed check"
    bank = "Bank transfer (automatic)"
    card = "Credit card (automatic)"


class CustomerFeatures(BaseModel):
    """One customer record. Every field is required and type/enum validated."""

    gender: Gender = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes", examples=[0])
    Partner: YesNo = Field(..., examples=["No"])
    Dependents: YesNo = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, le=120, description="Months with the company", examples=[2])
    PhoneService: YesNo = Field(..., examples=["Yes"])
    MultipleLines: MultipleLinesOption = Field(..., examples=["No"])
    InternetService: InternetServiceType = Field(..., examples=["Fiber optic"])
    OnlineSecurity: InternetAddOn = Field(..., examples=["No"])
    OnlineBackup: InternetAddOn = Field(..., examples=["No"])
    DeviceProtection: InternetAddOn = Field(..., examples=["No"])
    TechSupport: InternetAddOn = Field(..., examples=["No"])
    StreamingTV: InternetAddOn = Field(..., examples=["Yes"])
    StreamingMovies: InternetAddOn = Field(..., examples=["Yes"])
    Contract: ContractType = Field(..., examples=["Month-to-month"])
    PaperlessBilling: YesNo = Field(..., examples=["Yes"])
    PaymentMethod: PaymentMethodType = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, le=1000, examples=[94.40])
    TotalCharges: Optional[float] = Field(
        None, ge=0, description="Blank/null is accepted for brand-new customers "
                                "(reconstructed as tenure x MonthlyCharges).",
        examples=[188.80],
    )

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        """The source data encodes 'no billing yet' as a blank string — accept it."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "Yes", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
                "MonthlyCharges": 94.40, "TotalCharges": 188.80,
            }
        }
    }


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


# --------------------------------------------------------------------------- #
# Response schema
# --------------------------------------------------------------------------- #
class PredictionResponse(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float
    risk_level: Literal["Low", "Medium", "High", "Very High"]
    threshold_used: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    count: int
    predicted_churners: int
    results: List[PredictionResponse]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_model() -> None:
    if "pipeline" not in ARTIFACT:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run `python train_model.py` and restart the service.",
        )


def _risk_level(p: float) -> str:
    if p < 0.30:
        return "Low"
    if p < 0.50:
        return "Medium"
    if p < 0.75:
        return "High"
    return "Very High"


def _score(frame: pd.DataFrame) -> List[PredictionResponse]:
    """Run the pipeline and wrap the output in response objects."""
    pipeline = ARTIFACT["pipeline"]
    threshold = float(ARTIFACT.get("threshold", 0.5))
    version = str(ARTIFACT.get("model_version", "unknown"))

    probs = pipeline.predict_proba(frame)[:, 1]
    return [
        PredictionResponse(
            prediction="Yes" if p >= threshold else "No",
            churn_probability=round(float(p), 4),
            risk_level=_risk_level(float(p)),
            threshold_used=threshold,
            model_version=version,
        )
        for p in probs
    ]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Telco Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["POST /predict", "POST /predict/batch", "GET /health", "GET /model-info"],
    }


@app.get("/health", tags=["meta"])
def health():
    """Liveness probe — reports whether the model artifact is loaded."""
    loaded = "pipeline" in ARTIFACT
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "model_version": ARTIFACT.get("model_version"),
        "trained_at": ARTIFACT.get("trained_at"),
    }


@app.get("/model-info", tags=["meta"])
def model_info():
    """Model card: parameters, hold-out metrics and the expected input schema."""
    _require_model()
    return {
        "model_type": "DecisionTreeClassifier (inside a scikit-learn Pipeline)",
        "model_version": ARTIFACT.get("model_version"),
        "trained_at": ARTIFACT.get("trained_at"),
        "sklearn_version": ARTIFACT.get("sklearn_version"),
        "hyperparameters": ARTIFACT.get("model_params"),
        "decision_threshold": ARTIFACT.get("threshold"),
        "holdout_metrics": ARTIFACT.get("metrics"),
        "required_input_fields": ARTIFACT.get("input_columns"),
        "engineered_features": cf.ENGINEERED_NUMERIC + cf.ENGINEERED_CATEGORICAL,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: CustomerFeatures):
    """Predict churn for a single customer.

    Returns the Yes/No decision, the churn probability, and a risk band.
    Invalid payloads are rejected by Pydantic with HTTP 422 and a field-level
    explanation before the model is ever called.
    """
    _require_model()
    try:
        frame = pd.DataFrame([customer.model_dump()])
        result = _score(frame)[0]
        logger.info("prediction=%s p=%.4f tenure=%s contract=%s",
                    result.prediction, result.churn_probability,
                    customer.tenure, customer.Contract.value)
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal error.",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(payload: BatchRequest):
    """Score up to 1,000 customers in a single call (bonus endpoint).

    Useful for the nightly job that refreshes the retention team's call list.
    """
    _require_model()
    try:
        frame = pd.DataFrame([c.model_dump() for c in payload.customers])
        results = _score(frame)
        return BatchPredictionResponse(
            count=len(results),
            predicted_churners=sum(r.prediction == "Yes" for r in results),
            results=results,
        )
    except Exception:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed due to an internal error.",
        )


@app.exception_handler(404)
async def not_found(request, exc):  # pragma: no cover
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. See /docs for the available endpoints."},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
