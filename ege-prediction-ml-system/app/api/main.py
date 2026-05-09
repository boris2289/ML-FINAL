from fastapi import FastAPI, HTTPException

from app.api.schemas import PredictionRequest, PredictionResponse
from app.api.services import ModelNotReadyError, PredictorService

app = FastAPI(title="EGE Prediction API", version="1.0.0")
predictor = PredictorService()


@app.get("/")
def root() -> dict:
    return {
        "message": "EGE Prediction API is running",
        "model_loaded": predictor.is_ready,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": predictor.is_ready,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        features = request.model_dump()
        return PredictionResponse(**predictor.predict(features))
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
