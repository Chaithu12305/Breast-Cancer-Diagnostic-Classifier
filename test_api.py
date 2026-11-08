from fastapi.testclient import TestClient

from bcclf.api import app
from bcclf.data import load_dataset


def test_api_end_to_end(trained_env):
    X, y, _ = load_dataset()
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert len(client.get("/features").json()) == 30
        idx = int(y[y == 0].index[0])  # a known malignant sample
        r = client.post("/predict", json={"features": X.loc[idx].to_dict()})
        assert r.status_code == 200
        assert r.json()["prediction"] == "malignant"
        bad = client.post("/predict", json={"features": {"mean radius": 1.0}})
        assert bad.status_code == 422
