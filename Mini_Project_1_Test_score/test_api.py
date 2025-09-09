import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Create a Test Client instance for making requests."""
    with TestClient(app) as client:
        yield client

# test home route
def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html>" in response.text

def test_predict_page(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html>" in response.text

def test_predict_invalid_input(client):
    response = client.post("/predict", json={"hours_studied": -2.0})
    assert response.status_code == 400
    assert "Hours studied should be positive" in response.json()["detail"]