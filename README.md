# Term-Deposit Lead Scoring

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
  
## Data

The dataset is not stored in this repo. Download it from UCI:

1. Go to https://archive.ics.uci.edu/dataset/222/bank+marketing and download the zip.
2. Unzip it, then unzip the inner `bank.zip` to get `bank-full.csv`.
3. Put `bank-full.csv` in a `data/` folder (git-ignored), or in Google Drive if you use Colab.
4. Update `path` in the notebook's loading cell to match.

Citation: Moro, S., Cortez, P., Rita, P. (2014). A data-driven approach to predict the
success of bank telemarketing. Decision Support Systems.

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

## Benchmark results

All numbers come from `benchmark/bench.py`: 2,000 requests per concurrency level after a 100-request warm-up, with random but valid customers, over HTTP with JSON. Everything ran on a single 2-core GitHub Codespace, with Triton, the gateway and the load generator all sharing the same CPU. **Treat the numbers as relative comparisons, not absolute performance.** Each figure is from a single run.

### Triton directly: dynamic batching on vs. off

| Model | Concurrency | Req/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| batching on | 1 | 127.8 | 7.06 | 11.70 | 17.19 |
| batching off | 1 | 355.2 | 2.50 | 4.93 | 8.80 |
| batching on | 8 | 451.0 | 15.33 | 33.18 | 54.60 |
| batching off | 8 | 322.5 | 15.09 | 69.51 | 117.70 |
| batching on | 32 | 219.2 | 86.65 | 459.01 | 696.66 |
| batching off | 32 | 212.1 | 98.22 | 452.75 | 687.12 |
| batching on | 64 | 187.0 | 227.58 | 991.28 | 1577.81 |
| batching off | 64 | 197.9 | 223.99 | 924.41 | 1391.66 |

What this shows:

- **Low load (1 request at a time):** dynamic batching adds about 4.5 ms of latency. This is consistent with the 5 ms `max_queue_delay_microseconds` setting, because Triton waits briefly for other requests that never arrive.
- **Moderate load (8 concurrent):** batching gives about 40% more throughput and a much better tail (p99 55 ms vs. 118 ms).
- **High load (32 and 64):** both configurations are saturated and perform about the same. On this 2-core machine the CPU is the limit, so the differences here are within what I would expect from run-to-run noise.

### Through the FastAPI gateway (`/score`, batching on)

| Concurrency | Req/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|
| 1 | 69.7 | 13.21 | 23.49 | 31.34 |
| 8 | 114.5 | 61.68 | 132.58 | 186.49 |
| 32 | 76.7 | 281.52 | 1245.41 | 1932.99 |
| 64 | 55.7 | 760.72 | 3409.72 | 5069.83 |

The gateway adds validation, feature encoding and an extra HTTP hop, so it is slower than calling Triton directly, and it degrades faster under load. Part of the cost is Python and JSON overhead, and part is the three processes sharing two cores. I have not profiled to separate the two.

### Reproduce

```bash
python3 benchmark/bench.py --target triton --models lead_scorer lead_scorer_nobatch --json benchmark/results_triton.json
python3 benchmark/bench.py --target gateway --json benchmark/results_gateway.json
```

Raw results are in `benchmark/results_*.json`.

## Next steps

1. FastAPI gateway
2. vLLM explanation service
3. Async benchmark with and without dynamic batching
4. Optional: Kubernetes with `kind`


