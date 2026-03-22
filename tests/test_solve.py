from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_solve_customer_returns_completed(monkeypatch):
    def fake_execute_task(_client, _task):
        return {
            "kind": "customer",
            "result": {
                "id": 123,
                "name": "Testkunde AINM AS",
                "isCustomer": True,
                "email": "test@ainm.no",
            },
        }

    monkeypatch.setattr(main, "execute_task", fake_execute_task)

    response = client.post(
        "/solve",
        json={
            "prompt": 'Opprett kunde "Testkunde AINM AS" med e-post test@ainm.no',
            "files": [],
            "tripletex_credentials": {
                "base_url": "https://example.tripletex.dev/v2",
                "session_token": "dummy-token",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "completed"}

def test_solve_missing_tripletex_credentials_returns_422():
    response = client.post(
        "/solve",
        json={
            "prompt": 'Opprett kunde "Testkunde AINM AS" med e-post test@ainm.no',
            "files": [],
        },
    )

    assert response.status_code == 422

def test_solve_uses_tripletex_credentials_from_request(monkeypatch):
    captured = {}

    class FakeTripletexClient:
        def __init__(self, base_url, session_token):
            captured["base_url"] = base_url
            captured["session_token"] = session_token

    def fake_execute_task(_client, _task):
        return {"kind": "customer", "result": {"id": 123}}

    monkeypatch.setattr(main, "TripletexClient", FakeTripletexClient)
    monkeypatch.setattr(main, "execute_task", fake_execute_task)

    response = client.post(
        "/solve",
        json={
            "prompt": 'Opprett kunde "Testkunde AINM AS" med e-post test@ainm.no',
            "files": [],
            "tripletex_credentials": {
                "base_url": "https://real-sandbox.tripletex.dev/v2",
                "session_token": "real-token-for-test",
            },
        },
    )

    assert response.status_code == 200
    assert captured["base_url"] == "https://real-sandbox.tripletex.dev/v2"
    assert captured["session_token"] == "real-token-for-test"
