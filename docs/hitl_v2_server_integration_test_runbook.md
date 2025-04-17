# HITL V2 Server - Manual Integration Test Runbook

This document outlines the steps to manually test the core functionality of the HITL V2 backend server using `curl`.

## 1. Prerequisites

-   **Conda Environment:** Ensure the `ppa-agent` conda environment is activated.
    ```bash
    conda activate ppa-agent
    ```
-   **Poetry Dependencies:** Make sure all dependencies are installed.
    ```bash
    poetry install
    ```
-   **Environment Variables:** A `.env` file must exist in the project root (`/home/tsong/workspace/python/agenty`) containing necessary variables, especially `GOOGLE_API_KEY`.

## 2. Start the Server

Run the FastAPI server using Uvicorn from the project root directory. The `--reload` flag allows the server to restart automatically on code changes.

```bash
conda run -n ppa-agent poetry run uvicorn server:app --reload --port 8000
```

Keep this terminal window open to view server logs.

## 3. Test Case: Basic PPA Information Gathering Workflow

This test case simulates a customer providing information in multiple steps, requiring agent review.

**Note:** Replace `<THREAD_ID>` in subsequent steps with the actual `thread_id` returned by the `/threads/start` command.

### Step 3.1: Send Initial Email (Quote Request)

Open a **new terminal window** (leave the server running in the first one).

```bash
# Send the first email and capture the thread_id
THREAD_ID=$(curl -s -X POST http://localhost:8000/threads/start \
-H "Content-Type: application/json" \
-d '{
  "email_content": "Subject: Need Auto Insurance Quote\\n\\nHi,\\n\\nI need a quote for my car insurance.\\n\\nThanks,\\nJohn"
}' | jq -r '.thread_id')

# Print the thread_id (optional)
echo "Started Thread ID: ${THREAD_ID}"
```

*   **Expected Outcome:** The command should return a JSON object containing a `thread_id`. The server logs should show processing and potentially indicate `status: 'new'` or `info_incomplete` depending on the initial assessment.

### Step 3.2: Send Follow-up Email (Name, Age, Address)

Use the `THREAD_ID` obtained in the previous step.

```bash
curl -s -X POST http://localhost:8000/threads/${THREAD_ID}/process \
-H "Content-Type: application/json" \
-d '{
  "email_content": "Subject: Re: Need Auto Insurance Quote\\n\\nHi,\\n\\nMy name is Jane Roe, I am 42 years old, and I live at 456 Oak Ave, Sometown, CA 90211.\\n\\nThanks,\\nJane"
}' | jq
```

*   **Expected Outcome:** The command should return a JSON object with `status: 'info_incomplete'` and `review_required: true`, likely for the `generate_info_request` step. The `agent_response` should be `null` because review is needed.
*   **Check Logs (Optional):** Review `agent.log` to see the extracted `customer_info` and the state being saved.

### Step 3.3: Approve Review (Accept Info Request Message)

Simulate a human agent accepting the generated message asking for missing vehicle info.

```bash
curl -s -X POST http://localhost:8000/threads/${THREAD_ID}/review \
-H "Content-Type: application/json" \
-d '{
  "decision": "accepted",
  "feedback": null
}' | jq
```

*   **Expected Outcome:** The command should return a JSON object with `status: 'info_requested'`, `review_required: false`, and the `agent_response` containing the actual message requesting vehicle details.
*   **Check Logs (Optional):** Review `agent.log` to confirm the `resume_after_review` function executed correctly.

### Step 3.4: Send Final Email (Vehicle Details)

Provide the remaining required information (vehicle details).

```bash
curl -s -X POST http://localhost:8000/threads/${THREAD_ID}/process \
-H "Content-Type: application/json" \
-d '{
  "email_content": "Subject: Re: Need Auto Insurance Quote\\n\\nHi again,\\n\\nIt\'s a 2023 Toyota Camry.\\n\\nThanks,\\nJane"
}' | jq
```

*   **Expected Outcome:** The command should return a JSON object. The exact `status` will depend on the next workflow step. If all information is gathered, it might be `quote_ready` or similar, potentially requiring review for the generated quote (`generate_quote` step).

### Step 3.5: Further Steps (Optional)

-   If another review is required (e.g., for the quote), send another request to the `/review` endpoint with the appropriate decision (`accepted` or `rejected`).
-   Continue sending emails via the `/process` endpoint as needed to simulate the full conversation.

## 4. Checking State (Advanced / Debugging)

While testing, you can inspect the `agent.log` file for detailed state information logged by the `PPAAgent` during processing and review steps. This helps verify that information is extracted and persisted correctly.
