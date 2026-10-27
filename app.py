"""FastAPI service for Iris predictions and cosine similarity."""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from sklearn.datasets import load_iris


MODEL_PATH = Path(__file__).with_name("iris_logistic_pipeline.joblib")
iris = load_iris()

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model file not found: {MODEL_PATH}. "
        "Run `python logistic_regression_iris.py` first."
    )

model = joblib.load(MODEL_PATH)
app = FastAPI(title="Iris prediction API", version="1.0.0")


class IrisFeatures(BaseModel):
    """Measurements of one Iris flower, in centimeters."""

    sepal_length: float = Field(gt=0, description="Sepal length in centimeters")
    sepal_width: float = Field(gt=0, description="Sepal width in centimeters")
    petal_length: float = Field(gt=0, description="Petal length in centimeters")
    petal_width: float = Field(gt=0, description="Petal width in centimeters")


class VectorPair(BaseModel):
    """Two numeric vectors whose lengths must match."""

    vector_a: list[float]
    vector_b: list[float]

    @model_validator(mode="after")
    def vectors_have_same_length(self) -> "VectorPair":
        if len(self.vector_a) != len(self.vector_b):
            raise ValueError("vector_a and vector_b must have the same length")
        if not self.vector_a:
            raise ValueError("vectors must not be empty")
        return self


@app.post("/predict")
def predict(features: IrisFeatures) -> dict[str, str | int]:
    """Predict the Iris species for one set of measurements."""

    values = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    class_index = int(model.predict(values)[0])
    return {"class_index": class_index, "predicted_class": str(iris.target_names[class_index])}


@app.post("/similarity")
def similarity(vectors: VectorPair) -> dict[str, float]:
    """Return the cosine similarity of two vectors."""

    vector_a = np.asarray(vectors.vector_a, dtype=float)
    vector_b = np.asarray(vectors.vector_b, dtype=float)
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        raise HTTPException(status_code=422, detail="Cosine similarity is undefined for a zero vector")

    score = float(np.dot(vector_a, vector_b) / denominator)
    return {"cosine_similarity": score}
