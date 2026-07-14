"""
FastAPI - Service de prédiction d'effort à partir du titre + description d'une tâche.
Lancer avec : uvicorn main:app --reload --port 8001
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="SmartPM - Effort Estimation API")

# Correction : utilisation de __file__ (avec deux underscores de chaque côté)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "nlp_effort_model.pkl")
model = joblib.load(MODEL_PATH)

class TaskInput(BaseModel):
    title: str
    description: str

class TaskOutput(BaseModel):
    predicted_effort_hours: float

@app.get("/")
def health_check():
    return {"status": "ok", "message": "SmartPM Effort Estimation API is running"}

@app.post("/predict", response_model=TaskOutput)
def predict(data: TaskInput):
    if not data.title.strip() and not data.description.strip():
        raise HTTPException(status_code=400, detail="Titre ou description requis")
    
    text = f"{data.title} {data.description}".strip()
    prediction = model.predict([text])[0]
    prediction = max(0, round(float(prediction), 2))
    
    return TaskOutput(predicted_effort_hours=prediction)