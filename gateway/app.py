"""FastAPI gateway in front of NVIDIA Triton.

Accepts a customer as normal JSON, builds the feature vector the model was trained on,
asks Triton for the probability of "yes", and applies the tuned decision threshold.
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "lead_scorer")
DEFAULT_METADATA = Path(__file__).resolve().parent.parent / "model_repository" / "metadata.json"
METADATA_PATH = Path(os.getenv("METADATA_PATH", DEFAULT_METADATA))
MAX_BATCH = 64  # keep in sync with max_batch_size in config.pbtxt


# Optional LLM explanation service (vLLM, OpenAI-compatible API). /explain needs it; /score does not.
VLLM_URL = os.getenv("VLLM_URL", "").rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
VLLM_MODEL = os.getenv("VLLM_MODEL", "explainer")

# ---- Feature layout, loaded from the file saved by the training notebook ----------------
_meta = json.loads(METADATA_PATH.read_text())
FEATURE_NAMES: list[str] = _meta["feature_names"]
THRESHOLD: float = float(_meta["threshold"])
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)

NUMERIC_FIELDS = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_FIELDS = [
    "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome",
]
# One-hot columns are named "<field>_<value>"; this recovers the values the model knows.
CATEGORIES = {
    field: sorted(n[len(field) + 1:] for n in FEATURE_NAMES if n.startswith(field + "_"))
    for field in CATEGORICAL_FIELDS
}

# Fail at startup, not at request time, if the metadata does not match this schema.
assert all(f in FEATURE_INDEX for f in NUMERIC_FIELDS), "numeric feature missing from metadata"
assert len(NUMERIC_FIELDS) + sum(len(v) for v in CATEGORIES.values()) == N_FEATURES, (
    "metadata.json has features this gateway does not know how to build"
)


class Customer(BaseModel):
    """One customer. `duration` is deliberately not accepted: it is only known after the call."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 41, "job": "management", "marital": "married", "education": "tertiary",
                "default": "no", "balance": 1500, "housing": "no", "loan": "no",
                "contact": "cellular", "day": 15, "month": "mar", "campaign": 1,
                "pdays": -1, "previous": 0, "poutcome": "unknown",
            }
        },
    )

    age: int = Field(ge=18, le=120)
    job: str
    marital: str
    education: str
    default: str = "no"
    balance: int
    housing: str
    loan: str
    contact: str = "unknown"
    day: int = Field(ge=1, le=31)
    month: str
    campaign: int = Field(default=1, ge=1)
    pdays: int = -1
    previous: int = Field(default=0, ge=0)
    poutcome: str = "unknown"


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class ScoreResponse(BaseModel):
    probability: float
    threshold: float
    decision: str  # "call" or "skip"
    triton_ms: float


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    triton_ms: float



class ExplainResponse(ScoreResponse):
    explanation: str | None  # None if the LLM service is not configured or failed
    llm_ms: float | None = None
    llm_error: str | None = None

def encode(customer: Customer) -> list[float]:
    """Customer record -> the same ordered, one-hot-encoded numbers used in training."""
    row = [0.0] * N_FEATURES
    for field in NUMERIC_FIELDS:
        row[FEATURE_INDEX[field]] = float(getattr(customer, field))
    for field in CATEGORICAL_FIELDS:
        value = getattr(customer, field).strip().lower()
        column = f"{field}_{value}"
        if column not in FEATURE_INDEX:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown value '{value}' for '{field}'. Allowed: {CATEGORIES[field]}",
            )
        row[FEATURE_INDEX[column]] = 1.0
    return row


async def triton_ready(client: httpx.AsyncClient) -> bool:
    try:
        return (await client.get("/v2/health/ready")).status_code == 200
    except httpx.HTTPError:
        return False


async def run_model(client: httpx.AsyncClient, rows: list[list[float]]) -> list[float]:
    """Send up to MAX_BATCH rows to Triton in one request; return P(yes) for each row."""
    payload = {
        "inputs": [{
            "name": "input",
            "shape": [len(rows), N_FEATURES],
            "datatype": "FP32",
            "data": [x for row in rows for x in row],
        }]
    }
    try:
        resp = await client.post(f"/v2/models/{MODEL_NAME}/infer", json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Triton unreachable: {exc!r}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Triton error {resp.status_code}: {resp.text[:300]}")
    outputs = {o["name"]: o for o in resp.json()["outputs"]}
    return outputs["probabilities"]["data"][1::2]  # each row is [P(no), P(yes)]


async def score_rows(client: httpx.AsyncClient, rows: list[list[float]]) -> list[float]:
    probs: list[float] = []
    for start in range(0, len(rows), MAX_BATCH):
        probs.extend(await run_model(client, rows[start:start + MAX_BATCH]))
    return probs


def to_response(prob: float, triton_ms: float) -> ScoreResponse:
    return ScoreResponse(
        probability=round(prob, 4),
        threshold=round(THRESHOLD, 4),
        decision="call" if prob >= THRESHOLD else "skip",
        triton_ms=round(triton_ms, 2),
    )


def build_messages(customer: Customer, result: ScoreResponse) -> list[dict]:
    """Prompt for the explanation. The LLM only sees the customer's details and the score."""
    facts = ", ".join(f"{k}={v}" for k, v in customer.model_dump().items())
    system = (
        "You write short notes for bank sales staff about term-deposit leads. "
        "You are given a customer's details and a model's score. "
        "Write exactly 2 plain-English sentences: describe the customer's profile using only the "
        "details given, then state the recommendation. "
        "Do not claim to know why the model produced its score. Do not invent facts."
    )
    user = (
        f"Customer: {facts}\n"
        f"Model score: {result.probability:.1%} chance of subscribing "
        f"(about 12% of customers overall subscribe; the call threshold is {result.threshold:.1%}).\n"
        f"Recommendation: {result.decision}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def call_llm(client: httpx.AsyncClient, messages: list[dict]) -> str:
    resp = await client.post("/v1/chat/completions", json={
        "model": VLLM_MODEL, "messages": messages, "max_tokens": 120, "temperature": 0.2,
    })
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=TRITON_URL, timeout=10.0)
    app.state.llm = (
        httpx.AsyncClient(base_url=VLLM_URL, headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                          timeout=60.0)
        if VLLM_URL else None
    )
    yield
    await app.state.client.aclose()
    if app.state.llm is not None:
        await app.state.llm.aclose()


app = FastAPI(title="Term-Deposit Lead Scoring Gateway", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    ready = await triton_ready(request.app.state.client)
    body = {"status": "ok" if ready else "triton_unavailable", "triton_ready": ready,
            "model": MODEL_NAME, "threshold": round(THRESHOLD, 4)}
    if not ready:
        raise HTTPException(status_code=503, detail=body)
    return body


@app.post("/score", response_model=ScoreResponse)
async def score(customer: Customer, request: Request):
    row = encode(customer)
    t0 = time.perf_counter()
    prob = (await score_rows(request.app.state.client, [row]))[0]
    return to_response(prob, (time.perf_counter() - t0) * 1000)


@app.post("/explain", response_model=ExplainResponse)
async def explain(customer: Customer, request: Request):
    """Score the customer with Triton, then ask the LLM (vLLM) for a short plain-English note.

    Scoring never depends on the LLM: if it is down or not configured, you still get the score.
    """
    row = encode(customer)
    t0 = time.perf_counter()
    prob = (await score_rows(request.app.state.client, [row]))[0]
    result = to_response(prob, (time.perf_counter() - t0) * 1000)
    base = result.model_dump()

    llm = request.app.state.llm
    if llm is None:
        return ExplainResponse(**base, explanation=None, llm_error="VLLM_URL is not set")
    t1 = time.perf_counter()
    try:
        text = await call_llm(llm, build_messages(customer, result))
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        return ExplainResponse(**base, explanation=None, llm_error=f"{type(exc).__name__}: {exc}"[:200])
    return ExplainResponse(**base, explanation=text, llm_ms=round((time.perf_counter() - t1) * 1000, 2))


@app.post("/score/batch", response_model=BatchResponse)
async def score_batch(body: BatchRequest, request: Request):
    rows = [encode(c) for c in body.customers]
    t0 = time.perf_counter()
    probs = await score_rows(request.app.state.client, rows)
    elapsed = (time.perf_counter() - t0) * 1000
    return BatchResponse(results=[to_response(p, elapsed) for p in probs], triton_ms=round(elapsed, 2))