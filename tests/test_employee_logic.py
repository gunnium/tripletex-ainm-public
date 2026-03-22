from types import SimpleNamespace

import main


def test_ensure_employee_returns_existing_employee_without_creating(monkeypatch):
    existing_employee = {
        "id": 555,
        "firstName": "Ola",
        "lastName": "Nordmann",
        "email": "ola@example.no",
    }

    def fake_find_employee(_client, first_name, last_name, email):
        assert first_name == "Ola"
        assert last_name == "Nordmann"
        assert email == "ola@example.no"
        return existing_employee

    def fake_post(_path, _payload):
        raise AssertionError("post should not be called when employee already exists")

    fake_client = SimpleNamespace(post=fake_post)

    monkeypatch.setattr(main, "find_employee", fake_find_employee)

    result = main.ensure_employee(
        fake_client,
        first_name="Ola",
        last_name="Nordmann",
        email="ola@example.no",
    )

    assert result == existing_employee


def test_ensure_employee_creates_employee_when_missing(monkeypatch):
    created_employee = {
        "id": 556,
        "firstName": "Ola",
        "lastName": "Nordmann",
        "email": "ola@example.no",
    }

    def fake_find_employee(_client, first_name, last_name, email):
        assert first_name == "Ola"
        assert last_name == "Nordmann"
        assert email == "ola@example.no"
        return None

    def fake_post(path, payload):
        assert path == "/employee"
        assert payload == {
            "firstName": "Ola",
            "lastName": "Nordmann",
            "userType": "NO_ACCESS",
            "email": "ola@example.no",
        }
        return {"value": created_employee}

    fake_client = SimpleNamespace(post=fake_post)

    monkeypatch.setattr(main, "find_employee", fake_find_employee)

    result = main.ensure_employee(
        fake_client,
        first_name="Ola",
        last_name="Nordmann",
        email="ola@example.no",
    )

    assert result == created_employee
