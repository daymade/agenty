# PPA New Business AI Agent API

This document describes the HTTP API for interacting with the PPA New Business AI Agent.

## Prerequisites

- Start the server:
  ```bash
  conda run -n ppa-agent poetry run uvicorn server:app --reload --host 0.0.0.0 --port 8000
  ```
- Ensure your environment has `GEMINI_API_KEY` set if using the Gemini provider.

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### GET /health

Health check endpoint.

**Request**
```
GET /health HTTP/1.1
Host: localhost:8000
```

**Response**

```json
{
  "status": "ok"
}
```

---

### POST /start

Kick off a new conversation by processing the initial customer email.

**Request**
```
POST /start HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "email": "Hi, I would like a quote for my car Toyota Camry."
}
```

**Response**

```json
{
  "thread_id": "a9134584-377e-414a-8b5c-53aeb9506349",
  "state": {
    "thread_id": "a9134584-377e-414a-8b5c-53aeb9506349",
    "step_requiring_review": "analyze_information",
    "data_for_review": {
      "generated_intent_message": "..."
    },
    ...
  }
}
```

**curl Example**

```bash
curl -X POST http://localhost:8000/start \
  -H "Content-Type: application/json" \
  -d '{"email":"Hi, I would like a quote for my car Toyota Camry."}'
```

---

### POST /review

Resume the workflow after a human review decision.

**Request**
```
POST /review HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "thread_id": "a9134584-377e-414a-8b5c-53aeb9506349",
  "decision": "accepted",
  "feedback": "Optional feedback text"
}
```

**Response**

```json
{
  "thread_id": "a9134584-377e-414a-8b5c-53aeb9506349",
  "state": {
    "thread_id": "a9134584-377e-414a-8b5c-53aeb9506349",
    "step_requiring_review": "generate_info_request",
    "data_for_review": {
      "generated_info_request_message": "..."
    },
    ...
  }
}
```

**curl Example**

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"a9134584-377e-414a-8b5c-53aeb9506349","decision":"accepted"}'
```

---

## Integration Test Workflow

The following `curl` commands mirror the steps in `tests/test_agent_integration.py`:

```bash
# Turn 1: Initial Email -> Analyze Info Review
RESPONSE1=$(curl -s -X POST http://localhost:8000/start \
  -H "Content-Type: application/json" \
  -d '{"email":"Hi, I would like a quote for my car Toyota Camry."}')
echo "$RESPONSE1" | jq .

# Extract thread_id
THREAD_ID=$(echo "$RESPONSE1" | jq -r .thread_id)

# Turn 2: Accept Analyze Info -> Generate Info Request Review
RESPONSE2=$(curl -s -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d "{\"thread_id\":\"$THREAD_ID\",\"decision\":\"accepted\"}")
 echo "$RESPONSE2" | jq .

# Turn 3: Accept Generate Info Request -> Final State
curl -s -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d "{\"thread_id\":\"$THREAD_ID\",\"decision\":\"accepted\"}" \
  | jq .
```

> **Note:** install [`jq`](https://stedolan.github.io/jq/) for pretty JSON output.

---

You can extend this API with additional endpoints for other workflow steps, such as discount checks or quote generation, following the same patterns above.
