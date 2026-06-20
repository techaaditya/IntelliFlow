"""Smoke test for the IntelliFlow FastAPI AutoML routes."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris

from api.main import app


def main() -> None:
    client = TestClient(app)

    health = client.get("/health")
    print("health:", health.status_code, health.json())

    info = client.get("/automl/model-info")
    print("model-info:", info.status_code, info.json())

    iris = load_iris()
    row = dict(zip(iris.feature_names, iris.data[0].tolist()))
    prediction = client.post("/automl/predict", json={"rows": [row]})
    print("predict:", prediction.status_code, prediction.json())


if __name__ == "__main__":
    main()

