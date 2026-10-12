# Triton Lead Scoring Service

Serve a term-deposit lead-scoring model with **NVIDIA Triton Inference Server**
(ONNX Runtime backend, dynamic batching), put a **FastAPI** gateway in front of it,
and explain each score with an LLM served by **vLLM**. Everything runs in containers,
and a benchmark measures latency and throughput.

> Status: work in progress. Model training and ONNX export are done; the gateway,
> vLLM service and benchmark are next.

## Problem

A bank phones customers about a term deposit, and only about 12% say yes.
Given a customer's profile and history, predict the probability they subscribe,
so the best prospects can be called first.

- **Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
  (id 222), `bank-full.csv`, 45,211 rows x 17 columns, semicolon-separated.
  The data is not stored in this repo; download it from UCI.
- **Leakage note:** the `duration` column (call length) is dropped because it is only
  known after the call.

## Architecture

```
UCI Bank Marketing data
        |
scikit-learn GradientBoostingClassifier
        |  export (skl2onnx)
      ONNX model
        |
NVIDIA Triton (Docker, onnxruntime backend, dynamic batching)
        |
FastAPI gateway (one-hot encode, call Triton, apply threshold)
        |
vLLM service (plain-English explanation of each score)
        |
Async benchmark (p50/p95 latency, requests per second)
```

## Model results (held-out test split, 9,043 rows)

| Model | ROC AUC |
|---|---|
| Logistic Regression (baseline) | 0.772 |
| Random Forest | 0.792 |
| HistGradientBoosting | 0.803 |
| **GradientBoosting (chosen)** | **0.802** |

Decision threshold tuned for F1: **0.178** (precision 0.47, recall 0.53, F1 0.50),
compared with recall 0.20 at the default 0.5.
The ONNX export matches scikit-learn to within 3.9e-07.

**Limitations:** the threshold and model comparison used the test split, so the numbers
are slightly optimistic; a single train/test split was used.

## Repo layout

```
notebooks/01_train_and_export_model.ipynb   training, threshold tuning, ONNX export
model_repository/                           Triton model repository (lead_scorer + metadata.json)
gateway/                                    FastAPI gateway (todo)
llm_service/                                vLLM explanation service (todo)
benchmark/                                  async benchmark (todo)
```

## Run Triton

Requires Docker (GitHub Codespaces works well).

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $PWD/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models
```

Health check (expect `200`):

```bash
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/v2/health/ready
```

## Benchmark results

_To be added._

## Next steps

1. FastAPI gateway
2. vLLM explanation service
3. Async benchmark with and without dynamic batching
4. Optional: Kubernetes with `kind`
