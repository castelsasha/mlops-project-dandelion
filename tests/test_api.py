from fastapi.testclient import TestClient

# On importe l'app telle quelle (sans lancer de serveur)
from src.api.main import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    # on ne dépend pas d'un modèle présent pour que ce test soit robuste
    assert data.get("status") == "ok"
    assert "model_loaded" in data
