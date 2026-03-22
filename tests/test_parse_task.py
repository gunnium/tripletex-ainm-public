from main import parse_task


def test_parse_task_customer_norwegian():
    task = parse_task('Opprett kunde "Testkunde AINM AS" med e-post test@ainm.no')

    assert task["kind"] == "customer"
    assert task["name"] == "Testkunde AINM AS"
    assert task["email"] == "test@ainm.no"
