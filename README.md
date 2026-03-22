# AINM Tripletex Solver

Dette prosjektet er en FastAPI-basert løser for AINM Tripletex-oppgaven.

## Status

Prosjektet støtter i dag noen enkle tasktyper og har gitt score i konkurransen.
Løsningen er fortsatt under utvikling og dekker ikke alle taskfamilier.

Per nå er fokus og dekning best på:
- create customer
- create employee
- create project
- create invoice

Det er også påbegynt arbeid for mer komplekse workflows, men disse er ikke fullført.

## Teknologi

- Python
- FastAPI
- requests
- pytest
- Render for HTTPS deploy

## Lokal kjøring

Opprett virtualenv og installer avhengigheter:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
