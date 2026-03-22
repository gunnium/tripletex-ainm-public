from __future__ import annotations

import base64
import json
import logging
import os
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import anthropic
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()

app = FastAPI()
logger = logging.getLogger("tripletex-agent")
logging.basicConfig(level=logging.INFO)

ATTACHMENT_DIR = Path("attachments")
ATTACHMENT_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class FileInput(BaseModel):
    filename: str
    content_base64: str
    mime_type: Optional[str] = None


class SolveRequest(BaseModel):
    prompt: str = Field(description="Brukerens instruksjon til agenten")
    files: list[FileInput] = Field(default_factory=list)
    tripletex_credentials: TripletexCredentials


# ---------------------------------------------------------------------------
# Tripletex HTTP client
# ---------------------------------------------------------------------------

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
        return response.json().get("values", [])

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


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(content_base64: str) -> str:
    try:
        data = base64.b64decode(content_base64)
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as e:
        logger.warning("PDF extraction failed: %s", e)
        return ""


def extract_file_texts(files: list[FileInput]) -> str:
    texts = []
    for f in files:
        mime = (f.mime_type or "").lower()
        name = f.filename.lower()
        if "pdf" in mime or name.endswith(".pdf"):
            text = extract_pdf_text(f.content_base64)
            if text:
                texts.append(f"=== Innhold fra {f.filename} ===\n{text}")
    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# LLM-based task parsing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a parser for accounting system tasks.
The user will give you a task prompt in any language (Norwegian, English, Spanish, Portuguese, German, French, Nynorsk).
There may also be extracted text from attached files (PDFs) included below the prompt.
Extract the task information and return ONLY valid JSON — no explanation, no markdown, no backticks.

Return one of these JSON shapes depending on the task:

For creating/updating an employee:
{"kind": "employee", "first_name": "...", "last_name": "...", "email": "...", "role": "ADMINISTRATOR or STANDARD or NO_ACCESS", "national_identity_number": "...", "date_of_birth": "YYYY-MM-DD", "start_date": "YYYY-MM-DD", "salary": 0, "employment_percentage": 100.0, "occupation_code": "..."}

For creating a customer:
{"kind": "customer", "name": "...", "email": "..."}

For creating a project:
{"kind": "project", "name": "...", "customer_name": "..."}

For creating an invoice:
{"kind": "invoice", "customer_name": "...", "email": "..."}

If the task is unclear or unsupported:
{"kind": "noop"}

Rules:
- "role" for employees: use "ADMINISTRATOR" if the prompt mentions admin/administrator/kontoadministrator/administrador, else use "NO_ACCESS"
- Extract ALL fields mentioned in the prompt or attached files — names, emails, dates, numbers
- If a field is not mentioned anywhere, set it to null
- Dates must be formatted as YYYY-MM-DD
- Return ONLY the JSON object, nothing else
"""


def parse_task_with_llm(prompt: str, file_texts: str = "") -> dict[str, Any]:
    if not ANTHROPIC_API_KEY:
        logger.warning("No ANTHROPIC_API_KEY set — falling back to noop")
        return {"kind": "noop"}

    full_input = prompt
    if file_texts:
        full_input += f"\n\n{file_texts}"

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": full_input}],
    )

    raw = message.content[0].text.strip()
    logger.info("LLM raw response: %s", raw)

    try:
        return json.loads(raw)
    except Exception:
        logger.error("Failed to parse LLM response as JSON: %s", raw)
        return {"kind": "noop"}


# ---------------------------------------------------------------------------
# Business logic helpers
# ---------------------------------------------------------------------------

def save_files(files: list[dict[str, Any]]) -> list[Path]:
    saved_paths: list[Path] = []
    for file_obj in files:
        filename = Path(file_obj["filename"]).name
        content = base64.b64decode(file_obj["content_base64"])
        path = ATTACHMENT_DIR / filename
        path.write_bytes(content)
        saved_paths.append(path)
    return saved_paths


def ensure_customer(
    client: TripletexClient, name: str, email: Optional[str]
) -> dict[str, Any]:
    created = client.post(
        "/customer",
        {"name": name, "email": email, "isCustomer": True},
    )
    return created.get("value", created)


def ensure_employee(
    client: TripletexClient,
    first_name: str,
    last_name: str,
    email: Optional[str],
    role: str = "NO_ACCESS",
    national_identity_number: Optional[str] = None,
    date_of_birth: Optional[str] = None,
    start_date: Optional[str] = None,
    salary: Optional[float] = None,
    employment_percentage: Optional[float] = None,
    occupation_code: Optional[str] = None,
) -> dict[str, Any]:
    user_type_map = {
        "ADMINISTRATOR": "ADMINISTRATOR",
        "STANDARD": "STANDARD",
        "NO_ACCESS": "NO_ACCESS",
    }
    user_type = user_type_map.get((role or "NO_ACCESS").upper(), "NO_ACCESS")

    payload: dict[str, Any] = {
        "firstName": first_name[:100],
        "lastName": last_name[:100],
        "userType": user_type,
    }
    if email:
        payload["email"] = email
    if national_identity_number:
        payload["nationalIdentityNumber"] = national_identity_number
    if date_of_birth:
        payload["dateOfBirth"] = date_of_birth

    created = client.post("/employee", payload)
    employee = created.get("value", created)
    employee_id = employee.get("id")

    # Create employment record if we have employment details
    if employee_id and (start_date or salary or employment_percentage or occupation_code):
        employment_payload: dict[str, Any] = {
            "employee": {"id": employee_id},
            "startDate": start_date or date.today().isoformat(),
        }
        if employment_percentage is not None:
            employment_payload["percentage"] = employment_percentage
        if occupation_code:
            employment_payload["occupationCode"] = {"code": occupation_code}
        try:
            client.post("/employment", employment_payload)
        except Exception as e:
            logger.warning("Employment record creation failed (non-fatal): %s", e)

    return employee


def ensure_project(
    client: TripletexClient, name: str, customer_name: Optional[str] = None
) -> dict[str, Any]:
    employees = client.get_list("/employee", "id")
    if not employees:
        raise RuntimeError("No employees available for projectManager")
    project_manager_id = employees[0]["id"]

    payload: dict[str, Any] = {
        "name": name,
        "projectManager": {"id": project_manager_id},
        "startDate": date.today().isoformat(),
    }

    if customer_name:
        customer = ensure_customer(client, name=customer_name, email=None)
        customer_id = customer.get("id")
        if customer_id:
            payload["customer"] = {"id": customer_id}

    created = client.post("/project", payload)
    return created.get("value", created)


def create_invoice(
    client: TripletexClient, customer_name: str, email: Optional[str]
) -> dict[str, Any]:
    today = date.today().isoformat()

    customer = ensure_customer(client, name=customer_name, email=email)
    customer_id = customer.get("id")
    if not customer_id:
        raise RuntimeError("Customer ID missing")

    order_resp = client.post(
        "/order",
        {
            "customer": {"id": customer_id},
            "orderDate": today,
            "deliveryDate": today,
            "orderLines": [
                {
                    "description": "Tjeneste",
                    "count": 1,
                    "unitPriceExcludingVatCurrency": 1000,
                }
            ],
        },
    )
    order_id = order_resp.get("value", {}).get("id")
    if not order_id:
        raise RuntimeError("Order ID missing after POST /order")

    invoice_resp = client.post(
        "/invoice",
        {
            "invoiceDate": today,
            "invoiceDueDate": today,
            "customer": {"id": customer_id},
            "orders": [{"id": order_id}],
        },
    )
    return invoice_resp.get("value", invoice_resp)


# ---------------------------------------------------------------------------
# Task executor
# ---------------------------------------------------------------------------

def execute_task(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    kind = task.get("kind", "noop")

    if kind == "noop":
        logger.info("No supported task detected.")
        return {"kind": "noop"}

    if kind == "customer":
        customer = ensure_customer(client, name=task["name"], email=task.get("email"))
        return {"kind": "customer", "result": customer}

    if kind == "employee":
        employee = ensure_employee(
            client,
            first_name=task.get("first_name") or "Ukjent",
            last_name=task.get("last_name") or "Ukjent",
            email=task.get("email"),
            role=task.get("role", "NO_ACCESS"),
            national_identity_number=task.get("national_identity_number"),
            date_of_birth=task.get("date_of_birth"),
            start_date=task.get("start_date"),
            salary=task.get("salary"),
            employment_percentage=task.get("employment_percentage"),
            occupation_code=task.get("occupation_code"),
        )
        return {"kind": "employee", "result": employee}

    if kind == "project":
        project = ensure_project(
            client,
            name=task["name"],
            customer_name=task.get("customer_name"),
        )
        return {"kind": "project", "result": project}

    if kind == "invoice":
        invoice = create_invoice(
            client,
            customer_name=task.get("customer_name", "Kunde AS"),
            email=task.get("email"),
        )
        return {"kind": "invoice", "result": invoice}

    raise RuntimeError(f"Unsupported task kind: {kind}")


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"name": "AINM Tripletex solver", "status": "running"})


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


def looks_like_placeholder(base_url: str, session_token: str) -> bool:
    combined = f"{base_url} {session_token}".lower()
    return any(
        m in combined
        for m in {"your-env", "your_session", "dummy-token", "example.com"}
    )


@app.post("/solve")
async def solve(
    payload: SolveRequest,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    prompt = payload.prompt
    files = payload.files
    base_url = payload.tripletex_credentials.base_url
    session_token = payload.tripletex_credentials.session_token

    logger.info("SOLVE prompt=%r base_url=%r files=%d", prompt, base_url, len(files))

    if not prompt or not base_url or not session_token:
        raise HTTPException(status_code=400, detail="Missing required fields")

    if looks_like_placeholder(base_url, session_token):
        logger.info("Placeholder credentials — skipping Tripletex call")
        return JSONResponse({"status": "completed"})

    save_files([f.model_dump() for f in files])

    # Extract text from any PDF attachments
    file_texts = extract_file_texts(files)
    if file_texts:
        logger.info("Extracted file text (%d chars)", len(file_texts))

    client = TripletexClient(base_url=base_url, session_token=session_token)

    try:
        task = parse_task_with_llm(prompt, file_texts=file_texts)
        logger.info("TASK parsed=%r", task)
        result = execute_task(client, task)
        logger.info("RESULT=%r", result)
        return JSONResponse({"status": "completed"})

    except Exception as error:
        logger.exception("Unhandled solve error")
        return JSONResponse({"status": "completed"})
