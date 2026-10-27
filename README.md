# Iris Logistic Regression API

Small learning project: train a scikit-learn pipeline and expose it through FastAPI.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python logistic_regression_iris.py
uvicorn app:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the endpoints interactively.
The training script creates `iris_logistic_pipeline.joblib` beside the Python files; the API loads it at startup.

## Endpoints

### `POST /predict`

Request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response contains the predicted class index and species name.

### `POST /similarity`

Request:

```json
{
  "vector_a": [1, 2, 3],
  "vector_b": [2, 4, 6]
}
```

Both vectors must have the same non-zero length. A zero vector is rejected because cosine similarity is undefined for it.
