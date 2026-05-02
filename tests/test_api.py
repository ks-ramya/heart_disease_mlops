"""End-to-end API tests using FastAPI's TestClient."""

from __future__ import annotations

SAMPLE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


def test_root(api_client):
    r = api_client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"]
    assert body["model_loaded"] is True


def test_health_ok(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_predict_single(api_client):
    r = api_client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disease", "no_disease")
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {"disease", "no_disease"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_predict_batch(api_client):
    payload = {"instances": [SAMPLE, SAMPLE]}
    r = api_client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_validation_error(api_client):
    bad = {**SAMPLE, "age": -10}  # violates ge=0
    r = api_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_field(api_client):
    bad = {k: v for k, v in SAMPLE.items() if k != "age"}
    r = api_client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_batch_empty(api_client):
    r = api_client.post("/predict/batch", json={"instances": []})
    assert r.status_code == 400


def test_metrics_endpoint(api_client):
    """Prometheus /metrics should expose at least the default HTTP counters."""
    api_client.get("/health")  # generate one request to populate counters
    r = api_client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "http_requests_total" in text or "http_request_duration_seconds" in text


def test_ui_served(api_client):
    """Static UI is mounted at /ui/ and returns the index HTML."""
    r = api_client.get("/ui/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Heart Disease Risk" in r.text


def test_root_redirects_browsers(api_client):
    """Browsers (Accept: text/html) should be redirected to /ui/."""
    r = api_client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers.get("location", "").endswith("/ui/")
