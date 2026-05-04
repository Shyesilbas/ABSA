# Hybrid Backend Setup (C# Gateway + Python Inference)

This repo now supports:

- C# (`backend-csharp`) for API orchestration and optional client auth
- Python (`backend`) for ML inference and visualization logic

## Prerequisites

- .NET SDK 8+
- Python 3.10+ (recommended)
- Existing model file at `models/sentence_best_model.bin`

## 1) Run Python inference service (internal)

From project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set an internal token (optional but recommended):

```powershell
$env:INFERENCE_INTERNAL_TOKEN="dev-internal-token"
```

Run inference API:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## 2) Run C# gateway API (public)

Open `backend-csharp/appsettings.Development.json` and set:

- `Gateway.PythonBaseUrl` to `http://127.0.0.1:8000`
- `Gateway.InternalInferenceToken` to match `INFERENCE_INTERNAL_TOKEN`
- `Gateway.ClientApiKey` for external client auth (optional)

Run:

```bash
cd backend-csharp
dotnet restore
dotnet run
```

Gateway starts on `http://127.0.0.1:8001`.

## 3) Frontend behavior

Frontend is configured to call C# gateway by default:

- Vite proxy `/api` -> `http://127.0.0.1:8001`
- Production fallback in `frontend/src/api.ts` -> `http://127.0.0.1:8001`

If you set `Gateway.ClientApiKey`, include `x-api-key` on client requests.

## Notes

- `/health` and `/meta` stay publicly callable without client key for easy readiness checks.
- Inference routes are protected at Python level only when `INFERENCE_INTERNAL_TOKEN` is set.
- This is a migration-safe setup: ML code remains in Python while API orchestration moves to C#.
