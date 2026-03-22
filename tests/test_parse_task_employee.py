from main import parse_task


def test_parse_task_employee_norwegian():
    task = parse_task('Opprett ansatt "Ola Nordmann" med e-post ola@example.no')

    assert task["kind"] == "employee"
    assert task["first_name"] == "Ola"
    assert task["last_name"] == "Nordmann"
    assert task["email"] == "ola@example.no"
