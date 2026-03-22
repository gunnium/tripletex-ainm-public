import pytest

import main


def test_execute_task_invoice_creates_invoice_for_customer(monkeypatch):
    ensured_customer = {
        "id": 108246200,
        "name": "Testkunde AINM AS",
        "email": "test@ainm.no",
        "isCustomer": True,
    }

    created_invoice = {
        "id": 9001,
    }

    def fake_ensure_customer(_client, name, email):
        assert name == "Testkunde AINM AS"
        assert email == "test@ainm.no"
        return ensured_customer

    def fake_create_invoice(_client, customer_id):
        assert customer_id == 108246200
        return created_invoice

    monkeypatch.setattr(main, "ensure_customer", fake_ensure_customer)
    monkeypatch.setattr(main, "create_invoice", fake_create_invoice)

    result = main.execute_task(
        client=object(),
        task={
            "kind": "invoice",
            "customer_name": "Testkunde AINM AS",
            "email": "test@ainm.no",
        },
    )

    assert result["kind"] == "invoice"
    assert result["customer"] == ensured_customer
    assert result["result"] == created_invoice


def test_execute_task_invoice_raises_when_customer_id_missing(monkeypatch):
    def fake_ensure_customer(_client, name, email):
        return {
            "name": name,
            "email": email,
            "isCustomer": True,
        }

    monkeypatch.setattr(main, "ensure_customer", fake_ensure_customer)

    with pytest.raises(RuntimeError, match="Customer ID missing"):
        main.execute_task(
            client=object(),
            task={
                "kind": "invoice",
                "customer_name": "Testkunde AINM AS",
                "email": "test@ainm.no",
            },
        )
