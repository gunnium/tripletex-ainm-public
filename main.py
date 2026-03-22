
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI()
logger = logging.getLogger("tripletex-agent")
logging.basicConfig(level=logging.INFO)

ATTACHMENT_DIR = Path("attachments")
ATTACHMENT_DIR.mkdir(exist_ok=True)


class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class FileInput(BaseModel):
    filename: str
    content_base64: str


class SolveRequest(BaseModel):
    prompt: str = Field(description="Brukerens instruksjon til agenten")
    files: list[FileInput] = Field(default_factory=list, description="Vedlagte filer som base64")
    tripletex_credentials: TripletexCredentials = Field(
        description="Tripletex base_url og session_token"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": 'Opprett kunde "Testkunde AINM AS" med e-post test@ainm.no',
                "files": [],
                "tripletex_credentials": {
                    "base_url": "https://your-env.tripletex.dev/v2",
                    "session_token": "YOUR_SESSION_TOKEN",
                },
            }
        }
    }

class TaskResult(BaseModel):
    kind: str
    result: Optional[dict[str, Any]] = None
    customer: Optional[dict[str, Any]] = None


class SolveResponse(BaseModel):
    status: str
    result: TaskResult

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "completed",
                "result": {
                    "kind": "customer",
                    "result": {
                        "id": 108246200,
                        "name": "Testkunde AINM AS",
                        "isCustomer": True,
                        "email": "test@ainm.no"
                    }
                }
            }
        }
    }

class TripletexClient:
    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)

    def get_list(self, path: str, fields: str) -> list[dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}{path}",
            auth=self.auth,
            params={"fields": fields, "count": 100, "from": 0},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"Tripletex GET {path} failed: {response.status_code} {response.text}"
            )
        payload = response.json()
        return payload.get("values", [])

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            auth=self.auth,
            json=payload,
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"Tripletex POST {path} failed: {response.status_code} {response.text}"
            )
        return response.json()

    def put(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.put(
            f"{self.base_url}{path}",
            auth=self.auth,
            json=payload,
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"Tripletex PUT {path} failed: {response.status_code} {response.text}"
            )
        return response.json()

def save_files(files: list[dict[str, Any]]) -> list[Path]:
    saved_paths: list[Path] = []
    for file_obj in files:
        filename = Path(file_obj["filename"]).name
        content = base64.b64decode(file_obj["content_base64"])
        path = ATTACHMENT_DIR / filename
        path.write_bytes(content)
        saved_paths.append(path)
    return saved_paths


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else None


def extract_quoted_name(text: str) -> Optional[str]:
    for pattern in [r'"([^"]+)"', r"'([^']+)'"]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

def extract_name_after_keyword(text: str, keywords: set[str]) -> Optional[str]:
    cleaned = re.sub(
        r"\s+(med|with|con|com|mit|avec)\s+(e-post|epost|email)\s+[\w\.-]+@[\w\.-]+\.\w+.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    ordered_keywords = sorted(keywords, key=len, reverse=True)
    for keyword in ordered_keywords:
        pattern = rf"\b{re.escape(keyword)}\b\s+(?:for|til|para|pour|fur|für)?\s*['\"]?(.+?)['\"]?$"
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" ,.:;")
            if value:
                return value
    return None


def extract_person_name_after_keyword(text: str, keywords: set[str]) -> Optional[str]:
    cleaned = re.sub(
        r"\s+(med|with|con|com|mit|avec)\s+(e-post|epost|email)\s+[\w\.-]+@[\w\.-]+\.\w+.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    ordered_keywords = sorted(keywords, key=len, reverse=True)
    for keyword in ordered_keywords:
        pattern = rf"\b{re.escape(keyword)}\b\s+['\"]?(.+?)['\"]?$"
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" ,.:;")
            if value:
                return value
    return None

def contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)

def parse_task(prompt: str) -> dict[str, Any]:
    lower = normalize(prompt)
    quoted = extract_quoted_name(prompt)
    email = extract_email(prompt)

    invoice_keywords = {"invoice", "faktura", "factura", "fatura", "rechnung", "facture"}
    customer_keywords = {"customer", "kunde", "cliente", "client"}
    employee_keywords = {
        "employee",
        "ansatt",
        "tilsett",
        "empleado",
        "empregado",
        "funcionario",
        "mitarbeiter",
        "angestellt",
        "employe",
        "salarie",
    }
    project_keywords = {"project", "prosjekt", "proyecto", "projeto", "projekt", "projet"}

    if contains_any(lower, invoice_keywords):
        customer_name = quoted or extract_name_after_keyword(prompt, invoice_keywords) or "Acme AS"
        return {
            "kind": "invoice",
            "customer_name": customer_name,
            "email": email,
        }

    if contains_any(lower, customer_keywords):
        name = quoted or extract_name_after_keyword(prompt, customer_keywords) or "Acme AS"
        return {
            "kind": "customer",
            "name": name,
            "email": email,
        }

    if contains_any(lower, employee_keywords):
        full_name = quoted or extract_person_name_after_keyword(prompt, employee_keywords) or "Ola Nordmann"
        parts = full_name.split()
        first_name = parts[0]
        last_name = " ".join(parts[1:]) if len(parts) > 1 else "Nordmann"
        return {
            "kind": "employee",
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
        }

    if contains_any(lower, project_keywords):
        name = quoted or extract_name_after_keyword(prompt, project_keywords) or "Standardprosjekt"
        return {
            "kind": "project",
            "name": name,
        }

    return {"kind": "noop"}


def find_customer(
    client: TripletexClient, name: Optional[str], email: Optional[str]
) -> Optional[dict[str, Any]]:
    customers = client.get_list("/customer", "id,name,email,isCustomer")
    target_name = normalize(name) if name else None
    target_email = normalize(email) if email else None

    for customer in customers:
        if target_email and normalize(customer.get("email", "")) == target_email:
            return customer
        if target_name and normalize(customer.get("name", "")) == target_name:
            return customer
    return None


def ensure_customer(
    client: TripletexClient, name: str, email: Optional[str]
) -> dict[str, Any]:
    existing = find_customer(client, name=name, email=email)
    if existing:
        return existing

    created = client.post(
        "/customer",
        {
            "name": name,
            "email": email,
            "isCustomer": True,
        },
    )
    return created.get("value", created)

def find_employee(
    client: TripletexClient, first_name: str, last_name: str, email: Optional[str]
) -> Optional[dict[str, Any]]:
    employees = client.get_list("/employee", "id,firstName,lastName,email")
    full_name = normalize(f"{first_name} {last_name}")
    target_email = normalize(email) if email else None

    for employee in employees:
        employee_name = normalize(
            f"{employee.get('firstName', '')} {employee.get('lastName', '')}"
        )
        if target_email and normalize(employee.get("email", "")) == target_email:
            return employee
        if employee_name == full_name:
            return employee
    return None


def get_first_department(client: TripletexClient) -> Optional[int]:
    try:
        depts = client.get_list("/department", "id")
        if depts:
            return depts[0]["id"]
    except Exception:
        pass
    return None


def ensure_employee(
    client: TripletexClient, first_name: str, last_name: str, email: Optional[str]
) -> dict[str, Any]:
    existing = find_employee(client, first_name=first_name, last_name=last_name, email=email)
    if existing:
        return existing

    payload: dict[str, Any] = {
        "firstName": first_name,
        "lastName": last_name,
        "userType": "NO_ACCESS",
    }
    if email:
        payload["email"] = email

    created = client.post("/employee", payload)
    return created.get("value", created)

def find_project(client: TripletexClient, name: str) -> Optional[dict[str, Any]]:
    projects = client.get_list("/project", "id,name")
    target_name = normalize(name)
    for project in projects:
        if normalize(project.get("name", "")) == target_name:
            return project
    return None


def ensure_project(client: TripletexClient, name: str) -> dict[str, Any]:
    existing = find_project(client, name)
    if existing:
        return existing

    employees = client.get_list("/employee", "id,firstName,lastName,email")
    if not employees:
        raise RuntimeError("No employees available for projectManager")

    project_manager_id = employees[0]["id"]

    created = client.post(
        "/project",
        {
            "name": name,
            "projectManager": {"id": project_manager_id},
            "startDate": "2026-03-21",
        },
    )
    return created.get("value", created)

def create_invoice(client: TripletexClient, customer_id: int) -> dict[str, Any]:
    created = client.post(
        "/invoice",
        {
            "customer": {"id": customer_id},
            "invoiceDate": "2026-03-21",
            "invoiceDueDate": "2026-04-04",
            "orders": [
                {
                    "customer": {"id": customer_id},
                    "orderDate": "2026-03-21",
                    "deliveryDate": "2026-03-21",
                    "orderLines": [
                        {
                            "description": "Testlinje",
                            "count": 1,
                            "unitPriceExcludingVatCurrency": 100,
                        }
                    ],
                }
            ],
        },
    )
    return created.get("value", created)

def execute_task(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    kind = task["kind"]

    if kind == "noop":
        logger.info("No supported task detected.")
        return {"kind": "noop"}

    if kind == "customer":
        customer = ensure_customer(client, name=task["name"], email=task["email"])
        return {"kind": "customer", "result": customer}

    if kind == "employee":
        employee = ensure_employee(
            client,
            first_name=task["first_name"],
            last_name=task["last_name"],
            email=task["email"],
        )
        return {"kind": "employee", "result": employee}

    if kind == "project":
        project = ensure_project(client, name=task["name"])
        return {"kind": "project", "result": project}

    if kind == "invoice":
        customer = ensure_customer(
            client,
            name=task["customer_name"],
            email=task["email"],
        )
        customer_id = customer.get("id")
        if not customer_id:
            raise RuntimeError("Customer ID missing after ensure_customer")
        invoice = create_invoice(client, customer_id=customer_id)
        return {
            "kind": "invoice",
            "customer": customer,
            "result": invoice,
        }

    raise RuntimeError(f"Unsupported task kind: {kind}")

@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "name": "AINM Tripletex solver",
            "status": "running",
            "docs": "/docs",
            "healthz": "/healthz",
            "solve": "/solve",
        }
    )


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}

def looks_like_placeholder_tripletex_credentials(base_url: str, session_token: str) -> bool:
    combined = f"{base_url} {session_token}".lower()
    placeholder_markers = {
        "your-env.tripletex.dev",
        "your_session_token",
        "dummy-token",
        "example.tripletex.dev",
        "example.com",
    }
    return any(marker in combined for marker in placeholder_markers)


@app.post("/solve", response_model=SolveResponse)
async def solve(
    payload: SolveRequest,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    expected_bearer: Optional[str] = None

    if expected_bearer and authorization != f"Bearer {expected_bearer}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = payload.prompt
    files = [f.model_dump() for f in payload.files]
    base_url = payload.tripletex_credentials.base_url
    session_token = payload.tripletex_credentials.session_token

    logger.info("SOLVE prompt=%r base_url=%r", prompt, base_url)

    if not prompt or not base_url or not session_token:
        raise HTTPException(status_code=400, detail="Missing required fields")

    if looks_like_placeholder_tripletex_credentials(base_url, session_token):
        logger.info("Skipping Tripletex call — placeholder credentials")
        return JSONResponse({"status": "completed"})

    save_files(files)

    client = TripletexClient(base_url=base_url, session_token=session_token)

    try:
        task = parse_task(prompt)
        logger.info("TASK parsed=%r", task)
        result = execute_task(client, task)
        logger.info("RESULT=%r", result)
        return JSONResponse({"status": "completed"})

    except requests.HTTPError as error:
        logger.exception("Tripletex HTTP error")
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unhandled solve error")
        raise HTTPException(status_code=500, detail=str(error)) from error
