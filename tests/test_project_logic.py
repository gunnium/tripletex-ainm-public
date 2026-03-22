from types import SimpleNamespace

import main


def test_ensure_project_returns_existing_project_without_creating(monkeypatch):
    existing_project = {
        "id": 700,
        "name": "Nettside 2026",
    }

    def fake_find_project(_client, name):
        assert name == "Nettside 2026"
        return existing_project

    def fake_post(_path, _payload):
        raise AssertionError("post should not be called when project already exists")

    def fake_get_list(_path, _fields):
        raise AssertionError("get_list should not be called when project already exists")

    fake_client = SimpleNamespace(post=fake_post, get_list=fake_get_list)

    monkeypatch.setattr(main, "find_project", fake_find_project)

    result = main.ensure_project(fake_client, name="Nettside 2026")

    assert result == existing_project


def test_ensure_project_creates_project_when_missing(monkeypatch):
    created_project = {
        "id": 701,
        "name": "Nettside 2026",
    }

    def fake_find_project(_client, name):
        assert name == "Nettside 2026"
        return None

    def fake_get_list(path, fields):
        assert path == "/employee"
        assert fields == "id,firstName,lastName,email"
        return [
            {
                "id": 18470824,
                "firstName": "Gunnkristin",
                "lastName": "578c1323",
                "email": "gunnkristin@gmail.com",
            }
        ]

    def fake_post(path, payload):
        assert path == "/project"
        assert payload == {
            "name": "Nettside 2026",
            "projectManager": {"id": 18470824},
            "startDate": "2026-03-21",
        }
        return {"value": created_project}

    fake_client = SimpleNamespace(post=fake_post, get_list=fake_get_list)

    monkeypatch.setattr(main, "find_project", fake_find_project)

    result = main.ensure_project(fake_client, name="Nettside 2026")

    assert result == created_project
