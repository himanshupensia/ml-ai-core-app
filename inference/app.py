from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Inference API")

model = joblib.load("model.joblib")

class PredictionInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: PredictionInput):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {
        "prediction": int(prediction[0])
    }
