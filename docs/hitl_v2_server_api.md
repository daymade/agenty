# HITL V2 Backend Server API

## Overview

This document describes the API for the HITL (Human-in-the-Loop) V2 backend server. The server exposes the functionalities of the PPA New Business AI Agent, allowing interaction via HTTP requests to manage PPA insurance quoting workflows. It handles initiating new conversations, processing subsequent customer emails, and incorporating agency review decisions.

## Technology Stack

-   **Framework:** FastAPI
-   **Server:** Uvicorn
-   **Data Validation:** Pydantic

## Running the Server

To run the server locally, use the following command from the project root directory:

```bash
conda run -n ppa-agent poetry run uvicorn server:app --reload --port 8000
```

The server will be accessible at `http://127.0.0.1:8000`. FastAPI provides interactive documentation at `http://127.0.0.1:8000/docs`.

## API Endpoints

### 1. Start a New Conversation Thread

Initiates a new workflow thread with the first customer email.

-   **Endpoint:** `POST /threads/start`
-   **Request Body:** `application/json`
    ```json
    {
      "email_content": "string"
    }
    ```
-   **Response Body:** `application/json`
    ```json
    {
      "thread_id": "string",
      "agent_response": "object | string", // Depends on agent's output format
      "status": "string", // e.g., "awaiting_review", "completed", "error"
      "review_required": "boolean"
    }
    ```
-   **`curl` Example:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/threads/start" \
    -H "Content-Type: application/json" \
    -d '{
      "email_content": "Hi, I would like to get a quote for my 2023 Toyota Camry."
    }'
    ```

### 2. Process Subsequent Customer Email

Processes an email within an existing conversation thread.

-   **Endpoint:** `POST /threads/{thread_id}/process`
-   **Path Parameter:**
    -   `thread_id` (string): The ID of the existing conversation thread.
-   **Request Body:** `application/json`
    ```json
    {
      "email_content": "string"
    }
    ```
-   **Response Body:** `application/json`
    ```json
    {
      "thread_id": "string",
      "agent_response": "object | string",
      "status": "string",
      "review_required": "boolean"
    }
    ```
-   **`curl` Example:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/threads/some-thread-id-123/process" \
    -H "Content-Type: application/json" \
    -d '{
      "email_content": "My VIN is ABC123XYZ789."
    }'
    ```

### 3. Submit Agency Review Decision

Submits the human agent's decision after a review point.

-   **Endpoint:** `POST /threads/{thread_id}/review`
-   **Path Parameter:**
    -   `thread_id` (string): The ID of the conversation thread requiring review.
-   **Request Body:** `application/json`
    ```json
    {
      "decision": "string", // e.g., "approve", "reject", "request_clarification"
      "feedback": "string | null" // Optional feedback
    }
    ```
-   **Response Body:** `application/json`
    ```json
    {
      "thread_id": "string",
      "agent_response": "object | string", // Agent's response after incorporating feedback
      "status": "string", // e.g., "processing_next_step", "completed", "error"
      "review_required": "boolean" // Indicates if further review is needed
    }
    ```
-   **`curl` Example:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/threads/some-thread-id-123/review" \
    -H "Content-Type: application/json" \
    -d '{
      "decision": "approve",
      "feedback": "Looks good, proceed with generating the quote."
    }'
    ```

## Data Models (Pydantic)

The API uses the following Pydantic models (defined in `server.py`) for request and response validation:

-   `StartRequest`: Input for starting a new thread.
-   `StartResponse`: Output after starting a new thread.
-   `ProcessRequest`: Input for processing subsequent emails.
-   `ProcessResponse`: Output after processing subsequent emails.
-   `ReviewRequest`: Input for submitting a review decision.
-   `ReviewResponse`: Output after submitting a review decision.

These models ensure data integrity and provide clear contracts for API interaction.
