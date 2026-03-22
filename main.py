from __future__ import annotations

import base64
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Optional

import anthropic
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
# LLM-based task parsing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a parser for accounting system tasks.
The user will give you a task prompt in any language (Norwegian, English, Spanish, Portuguese, German, French, Nynorsk).
Extract the task information and return ONLY valid JSON — no explanation, no markdown, no backticks.

Return one of these JSON shapes depending on the task:

For creating/updating an employee:
{"kind": "employee", "first_name": "...", "last_name": "...", "email": "...", "role": "ADMINISTRATOR or STANDARD or NO_ACCESS"}

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
- All string values must be extracted exactly as written in the prompt (names, emails, etc.)
- If a field is not mentioned, set it to null
- Return ONLY the JSON object, nothing else
"""


def parse_task_with_llm(prompt: str) -> dict[str, Any]:
    if not ANTHROPIC_API_KEY:
        logger.warning("No ANTHROPIC_API_KEY set — falling back to noop")
        return {"kind": "noop"}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
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
) -> dict[str, Any]:
    user_type_map = {
        "ADMINISTRATOR": "ADMINISTRATOR",
        "STANDARD": "STANDARD",
        "NO_ACCESS": "NO_ACCESS",
    }
    user_type = user_type_map.get((role or "NO_ACCESS").upper(), "NO_ACCESS")

    payload: dict[str, Any] = {
        "firstName": first_name,
        "lastName": last_name,
        "userType": user_type,
    }
    if email:
        payload["email"] = email

    created = client.post("/employee", payload)
    return created.get("value", created)


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

    # Step 1: create customer
    customer = ensure_customer(client, name=customer_name, email=email)
    customer_id = customer.get("id")
    if not customer_id:
        raise RuntimeError("Customer ID missing")

    # Step 2: create order
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

    # Step 3: create invoice referencing the order
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
            first_name=task["first_name"],
            last_name=task["last_name"],
            email=task.get("email"),
            role=task.get("role", "NO_ACCESS"),
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
    files = [f.model_dump() for f in payload.files]
    base_url = payload.tripletex_credentials.base_url
    session_token = payload.tripletex_credentials.session_token

    logger.info("SOLVE prompt=%r base_url=%r", prompt, base_url)

    if not prompt or not base_url or not session_token:
        raise HTTPException(status_code=400, detail="Missing required fields")

    if looks_like_placeholder(base_url, session_token):
        logger.info("Placeholder credentials — skipping Tripletex call")
        return JSONResponse({"status": "completed"})

    save_files(files)
    client = TripletexClient(base_url=base_url, session_token=session_token)

    try:
        task = parse_task_with_llm(prompt)
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
