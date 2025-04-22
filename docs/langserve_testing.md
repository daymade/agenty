# Testing the PPA Agent via LangServe

This document describes how to test the PPA Agentic V2 using its LangServe API endpoints.

## Prerequisites

1.  **Activate Conda Environment:** Ensure the `ppa-agent` conda environment is activated:
    ```bash
    conda activate ppa-agent
    ```
2.  **Environment Variables:** Make sure necessary environment variables (like `GOOGLE_API_KEY` or `OPENAI_API_KEY`) are set.

## 1. Start the Server

Navigate to the project root directory (`/home/tsong/workspace/python/agenty`) and run the FastAPI server using the specific Python executable from your conda environment:

```bash
/home/tsong/miniconda3/envs/ppa-agent/bin/python -m uvicorn src.ppa_agentic_v2.server:app_fastapi --host 127.0.0.1 --port 8000 --reload
```

Keep this terminal running.

## 2. Interact using `curl`

You can interact with the agent using standard LangServe endpoints. The primary endpoint for sending input and getting output is `/ppa_agent_v2/invoke`.

Open a new terminal (with the conda environment activated, although not strictly necessary for `curl`).

**Example: Starting a New Conversation**

This command sends an initial message to start a conversation. We specify a `thread_id` within the `config.configurable` section for state management.

```bash
curl -X POST http://127.0.0.1:8000/ppa_agent_v2/invoke \
    -H 'Content-Type: application/json' \
    -d '{
          "input": {
            "messages": [
              {"type": "human", "content": "Hello, I need an auto insurance quote."}
            ]
          },
          "config": {
            "configurable": {
              "thread_id": "my-test-thread-123"
            }
          }
        }'
```

**Request Body:**

*   `input`: Contains the primary input for the graph. For this agent, it typically expects a dictionary with a `messages` key, holding a list of LangChain `BaseMessage` objects (or dictionaries representing them).
*   `config`: Contains configuration options, most importantly `configurable.thread_id` which is used by the checkpointer (`AsyncSqliteSaver`) to manage the conversation state.

**Example Response (Structure):**

The response contains the final state of the agent graph after processing the input.

```json
{
  "output": {
    // AgentState fields (goal, messages, customer_info, etc.)
    "thread_id": "...",
    "goal": "...",
    "messages": [ ... ],
    "customer_info": null,
    "mercury_session": null,
    "agent_scratchpad": "...",
    "planned_tool_inputs": null, 
    "action_pending_review": { ... }, // Populated if review is needed
    "requires_agency_review": true,
    "human_feedback": null,
    "is_waiting_for_customer": false,
    "event_history": [],
    "last_tool_outputs": null,
    "loop_count": 0
  },
  "metadata": {
    "run_id": "...", // LangServe run ID
    "feedback_tokens": []
  }
}
```

**Subsequent Turns:**

To continue the conversation, send another `POST` request to `/ppa_agent_v2/invoke` using the *same* `thread_id` in the config. The `input.messages` should contain the *new* message(s) for that turn.

```bash
curl -X POST http://127.0.0.1:8000/ppa_agent_v2/invoke \
    -H 'Content-Type: application/json' \
    -d '{
          "input": {
            "messages": [
              {"type": "human", "content": "My name is Bob, DOB 01/01/1990."}
            ]
          },
          "config": {
            "configurable": {
              "thread_id": "my-test-thread-123" // Same thread ID
            }
          }
        }'
```

## 3. Checking Thread State

This project includes a custom endpoint specifically for checking if a thread is awaiting review, which also implicitly confirms the thread exists and retrieves its latest planned action.

```bash
curl -X GET http://127.0.0.1:8000/threads/my-test-thread-123/status
```

**Example Response (Custom Endpoint):**

```json
{
  "thread_id": "my-test-thread-123",
  "awaiting_review": true,
  "planned_action": {
    "tool_name": "ask_customer_tool",
    "args": {
      "missing_fields": [
        "driver_name",
        "driver_dob",
        "address_line1",
        "city",
        "state_code",
        "zip_code"
      ]
    }
  },
  "error": null
}
```

This response confirms the thread exists and is currently paused awaiting review for the specified `ask_customer_tool` action.

## Other Endpoints

LangServe automatically provides other useful endpoints:

*   `/ppa_agent_v2/stream`: For streaming output.
*   `/ppa_agent_v2/batch`: For processing multiple inputs concurrently.
*   `/ppa_agent_v2/stream_log`: For streaming intermediate steps and logs.
*   `/docs`: FastAPI/Swagger UI for interactive API documentation.
*   `/redoc`: Alternative API documentation.

Refer to the LangServe documentation and the server's `/docs` endpoint for more details on these.
