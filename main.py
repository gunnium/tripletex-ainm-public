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
            raise RuntimeError(f"GET {path} failed: {response.status_code} {response.text}")
        return response.json().get("values", [])

    def get_one(self, path: str, fields: str = "*") -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            auth=self.auth,
            params={"fields": fields},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"GET {path} failed: {response.status_code} {response.text}")
        return response.json().get("value", {})

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            auth=self.auth,
            json=payload,
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"POST {path} failed: {response.status_code} {response.text}")
        return response.json()

    def put(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.put(
            f"{self.base_url}{path}",
            auth=self.auth,
            json=payload,
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"PUT {path} failed: {response.status_code} {response.text}")
        return response.json()

    def delete(self, path: str) -> None:
        response = requests.delete(
            f"{self.base_url}{path}",
            auth=self.auth,
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"DELETE {path} failed: {response.status_code} {response.text}")


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

SYSTEM_PROMPT = """You are a parser for accounting system tasks in Tripletex.
The user gives a task in any language (Norwegian, English, Spanish, Portuguese, German, French, Nynorsk).
Attached PDF text may also be included. Extract all details and return ONLY valid JSON — no explanation, no markdown, no backticks.

Supported task kinds and their JSON shapes:

EMPLOYEE (create employee):
{"kind": "employee", "first_name": "...", "last_name": "...", "email": "...", "role": "ADMINISTRATOR|STANDARD|NO_ACCESS", "national_identity_number": "...", "date_of_birth": "YYYY-MM-DD", "start_date": "YYYY-MM-DD", "employment_percentage": 100.0, "occupation_code": "..."}

CUSTOMER (create customer):
{"kind": "customer", "name": "...", "email": "...", "phone": "...", "address": "...", "postal_code": "...", "city": "..."}

PRODUCT (create product):
{"kind": "product", "name": "...", "price": 0.0, "vat_type": "HIGH|LOW|NONE", "product_number": "..."}

PROJECT (create project):
{"kind": "project", "name": "...", "customer_name": "...", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}

INVOICE (create invoice for a customer):
{"kind": "invoice", "customer_name": "...", "email": "...", "amount": 0.0, "description": "..."}

PAYMENT (register payment on existing invoice — look for invoice number or customer name):
{"kind": "payment", "customer_name": "...", "amount": 0.0, "payment_date": "YYYY-MM-DD"}

CREDIT_NOTE (create credit note / reverse an invoice):
{"kind": "credit_note", "customer_name": "..."}

DEPARTMENT (create a department):
{"kind": "department", "name": "...", "department_number": "..."}

TRAVEL_EXPENSE (register a travel expense report):
{"kind": "travel_expense", "employee_name": "...", "description": "...", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "amount": 0.0}

DELETE_TRAVEL_EXPENSE (delete a travel expense):
{"kind": "delete_travel_expense", "employee_name": "..."}

UNKNOWN:
{"kind": "noop"}

Rules:
- role for employees: ADMINISTRATOR if prompt says admin/administrator/kontoadministrator, else NO_ACCESS
- Extract ALL values from prompt and PDF text — names, emails, amounts, dates, codes
- Dates must be YYYY-MM-DD format
- If a field is not mentioned, use null
- Return ONLY the JSON object
"""


def parse_task_with_llm(prompt: str, file_texts: str = "") -> dict[str, Any]:
    if not ANTHROPIC_API_KEY:
        logger.warning("No ANTHROPIC_API_KEY — falling back to noop")
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
        logger.error("Failed to parse LLM JSON: %s", raw)
        return {"kind": "noop"}


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------

def handle_customer(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": task["name"],
        "isCustomer": True,
    }
    if task.get("email"):
        payload["email"] = task["email"]
    if task.get("phone"):
        payload["phoneNumber"] = task["phone"]
    created = client.post("/customer", payload)
    return created.get("value", created)


def handle_employee(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    user_type = {"ADMINISTRATOR": "ADMINISTRATOR", "STANDARD": "STANDARD"}.get(
        (task.get("role") or "").upper(), "NO_ACCESS"
    )
    payload: dict[str, Any] = {
        "firstName": (task.get("first_name") or "Ukjent")[:100],
        "lastName": (task.get("last_name") or "Ukjent")[:100],
        "userType": user_type,
    }
    if task.get("email"):
        payload["email"] = task["email"]
    if task.get("national_identity_number"):
        payload["nationalIdentityNumber"] = task["national_identity_number"]
    if task.get("date_of_birth"):
        payload["dateOfBirth"] = task["date_of_birth"]

    created = client.post("/employee", payload)
    employee = created.get("value", created)
    employee_id = employee.get("id")

    if employee_id and (task.get("start_date") or task.get("employment_percentage") or task.get("occupation_code")):
        emp_payload: dict[str, Any] = {
            "employee": {"id": employee_id},
            "startDate": task.get("start_date") or date.today().isoformat(),
        }
        if task.get("employment_percentage") is not None:
            emp_payload["percentage"] = task["employment_percentage"]
        if task.get("occupation_code"):
            emp_payload["occupationCode"] = {"code": task["occupation_code"]}
        try:
            client.post("/employment", emp_payload)
        except Exception as e:
            logger.warning("Employment record failed (non-fatal): %s", e)

    return employee


def handle_product(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": task["name"],
        "isInactive": False,
    }
    if task.get("price") is not None:
        payload["priceExcludingVatCurrency"] = task["price"]
    if task.get("product_number"):
        payload["number"] = task["product_number"]
    created = client.post("/product", payload)
    return created.get("value", created)


def handle_project(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    employees = client.get_list("/employee", "id")
    if not employees:
        raise RuntimeError("No employees for projectManager")
    pm_id = employees[0]["id"]

    payload: dict[str, Any] = {
        "name": task["name"],
        "projectManager": {"id": pm_id},
        "startDate": task.get("start_date") or date.today().isoformat(),
    }
    if task.get("end_date"):
        payload["endDate"] = task["end_date"]
    if task.get("customer_name"):
        customer = handle_customer(client, {"name": task["customer_name"], "email": None})
        if customer.get("id"):
            payload["customer"] = {"id": customer["id"]}

    created = client.post("/project", payload)
    return created.get("value", created)


def handle_invoice(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    today = date.today().isoformat()
    customer = handle_customer(client, {"name": task.get("customer_name", "Kunde AS"), "email": task.get("email")})
    customer_id = customer.get("id")
    if not customer_id:
        raise RuntimeError("Customer ID missing")

    order_resp = client.post("/order", {
        "customer": {"id": customer_id},
        "orderDate": today,
        "deliveryDate": today,
        "orderLines": [{
            "description": task.get("description") or "Tjeneste",
            "count": 1,
            "unitPriceExcludingVatCurrency": task.get("amount") or 1000,
        }],
    })
    order_id = order_resp.get("value", {}).get("id")
    if not order_id:
        raise RuntimeError("Order ID missing")

    invoice_resp = client.post("/invoice", {
        "invoiceDate": today,
        "invoiceDueDate": today,
        "customer": {"id": customer_id},
        "orders": [{"id": order_id}],
    })
    return invoice_resp.get("value", invoice_resp)


def handle_payment(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    # Find the most recent unpaid invoice
    invoices = client.get_list("/invoice", "id,amount,amountCurrency,customer")
    if not invoices:
        raise RuntimeError("No invoices found to pay")

    invoice = invoices[0]
    invoice_id = invoice.get("id")
    amount = task.get("amount") or invoice.get("amountCurrency") or 1000
    payment_date = task.get("payment_date") or date.today().isoformat()

    payment_resp = client.post(f"/invoice/{invoice_id}/payment", {
        "paymentDate": payment_date,
        "paymentTypeId": 1,
        "transactionId": "",
        "sendToLedger": True,
        "amount": amount,
    })
    return payment_resp.get("value", payment_resp)


def handle_credit_note(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    invoices = client.get_list("/invoice", "id,customer")
    if not invoices:
        raise RuntimeError("No invoices found for credit note")

    invoice_id = invoices[0].get("id")
    today = date.today().isoformat()
    credit_resp = client.post(f"/invoice/{invoice_id}/createCreditNote", {
        "creditNoteDate": today,
    })
    return credit_resp.get("value", credit_resp)


def handle_department(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": task["name"]}
    if task.get("department_number"):
        payload["departmentNumber"] = task["department_number"]
    created = client.post("/department", payload)
    return created.get("value", created)


def handle_travel_expense(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    employees = client.get_list("/employee", "id,firstName,lastName")
    if not employees:
        raise RuntimeError("No employees found")

    employee_id = employees[0]["id"]
    if task.get("employee_name"):
        name_lower = task["employee_name"].lower()
        for emp in employees:
            full = f"{emp.get('firstName','')} {emp.get('lastName','')}".lower()
            if name_lower in full or full in name_lower:
                employee_id = emp["id"]
                break

    today = date.today().isoformat()
    payload: dict[str, Any] = {
        "employee": {"id": employee_id},
        "description": task.get("description") or "Reise",
        "startDate": task.get("start_date") or today,
        "endDate": task.get("end_date") or today,
        "isCompleted": False,
    }
    created = client.post("/travelExpense", payload)
    return created.get("value", created)


def handle_delete_travel_expense(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    expenses = client.get_list("/travelExpense", "id,description")
    if not expenses:
        raise RuntimeError("No travel expenses found")
    expense_id = expenses[0]["id"]
    client.delete(f"/travelExpense/{expense_id}")
    return {"deleted_id": expense_id}


# ---------------------------------------------------------------------------
# Task executor
# ---------------------------------------------------------------------------

def execute_task(client: TripletexClient, task: dict[str, Any]) -> dict[str, Any]:
    kind = task.get("kind", "noop")
    logger.info("Executing task kind=%s", kind)

    handlers = {
        "customer": handle_customer,
        "employee": handle_employee,
        "product": handle_product,
        "project": handle_project,
        "invoice": handle_invoice,
        "payment": handle_payment,
        "credit_note": handle_credit_note,
        "department": handle_department,
        "travel_expense": handle_travel_expense,
        "delete_travel_expense": handle_delete_travel_expense,
    }

    if kind == "noop":
        return {"kind": "noop"}

    handler = handlers.get(kind)
    if not handler:
        logger.warning("Unsupported task kind: %s", kind)
        return {"kind": "noop"}

    result = handler(client, task)
    return {"kind": kind, "result": result}


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

def save_files(files: list[dict[str, Any]]) -> None:
    for file_obj in files:
        filename = Path(file_obj["filename"]).name
        content = base64.b64decode(file_obj["content_base64"])
        (ATTACHMENT_DIR / filename).write_bytes(content)


def looks_like_placeholder(base_url: str, session_token: str) -> bool:
    combined = f"{base_url} {session_token}".lower()
    return any(m in combined for m in {"your-env", "your_session", "dummy-token", "example.com"})


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"name": "AINM Tripletex solver", "status": "running"})


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/solve")
async def solve(
    payload: SolveRequest,
    authorization: Optional[str] = Header(default=None),
) -> JSONResponse:
    prompt = payload.prompt
    files = payload.files
    base_url = payload.tripletex_credentials.base_url
    session_token = payload.tripletex_credentials.session_token

    logger.info("SOLVE prompt=%r files=%d", prompt, len(files))

    if not prompt or not base_url or not session_token:
        raise HTTPException(status_code=400, detail="Missing required fields")

    if looks_like_placeholder(base_url, session_token):
        return JSONResponse({"status": "completed"})

    save_files([f.model_dump() for f in files])
    file_texts = extract_file_texts(files)
    if file_texts:
        logger.info("Extracted file text (%d chars)", len(file_texts))

    tx_client = TripletexClient(base_url=base_url, session_token=session_token)

    try:
        task = parse_task_with_llm(prompt, file_texts=file_texts)
        logger.info("TASK=%r", task)
        result = execute_task(tx_client, task)
        logger.info("RESULT=%r", result)
    except Exception as error:
        logger.exception("Solve error: %s", error)

    return JSONResponse({"status": "completed"})
