from main import parse_task


def test_parse_task_project_norwegian():
    task = parse_task('Opprett prosjekt "Nettside 2026"')

    assert task["kind"] == "project"
    assert task["name"] == "Nettside 2026"


def test_parse_task_invoice_norwegian():
    task = parse_task('Opprett faktura for "Testkunde AINM AS" med e-post test@ainm.no')

    assert task["kind"] == "invoice"
    assert task["customer_name"] == "Testkunde AINM AS"
    assert task["email"] == "test@ainm.no"
