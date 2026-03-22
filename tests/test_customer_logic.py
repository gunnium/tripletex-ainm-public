from types import SimpleNamespace

import main


def test_ensure_customer_returns_existing_customer_without_creating(monkeypatch):
    existing_customer = {
        "id": 108246200,
        "name": "Testkunde AINM AS",
        "isCustomer": True,
        "email": "test@ainm.no",
    }

    def fake_find_customer(_client, name, email):
        assert name == "Testkunde AINM AS"
        assert email == "test@ainm.no"
        return existing_customer

    def fake_post(_path, _payload):
        raise AssertionError("post should not be called when customer already exists")

    fake_client = SimpleNamespace(post=fake_post)

    monkeypatch.setattr(main, "find_customer", fake_find_customer)

    result = main.ensure_customer(
        fake_client,
        name="Testkunde AINM AS",
        email="test@ainm.no",
    )

    assert result == existing_customer


def test_ensure_customer_creates_customer_when_missing(monkeypatch):
    created_customer = {
        "id": 108246201,
        "name": "Ny Kunde AS",
        "isCustomer": True,
        "email": "ny@kunde.no",
    }

    def fake_find_customer(_client, name, email):
        assert name == "Ny Kunde AS"
        assert email == "ny@kunde.no"
        return None

    def fake_post(path, payload):
        assert path == "/customer"
        assert payload == {
            "name": "Ny Kunde AS",
            "email": "ny@kunde.no",
            "isCustomer": True,
        }
        return {"value": created_customer}

    fake_client = SimpleNamespace(post=fake_post)

    monkeypatch.setattr(main, "find_customer", fake_find_customer)

    result = main.ensure_customer(
        fake_client,
        name="Ny Kunde AS",
        email="ny@kunde.no",
    )

    assert result == created_customer
