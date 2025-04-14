# PPA New Business AI Agent - System Design

## 1. Introduction

### Goal
To design and implement a Proof of Concept (PoC) AI Agent using LangGraph to automate the initial stages of the Personal Private Auto (PPA) insurance new business quoting process within Mercury Insurance.

### Scope
This PoC focuses on:
*   Receiving and interpreting initial customer email inquiries for new PPA insurance.
*   Identifying the intent (new business) and line of business (PPA).
*   Extracting necessary information from the customer's email.
*   Identifying and requesting missing information required for a PPA quote.
*   Asking for proof of potential discounts.
*   Generating a draft quote and associated emails/requests.
*   Incorporating a Human-in-the-Loop (HITL) step where a human insurance agent reviews AI-generated communications before they are sent to the customer.
*   Handling basic state transitions based on information availability.

This PoC *does not* initially cover:
*   Direct integration with email servers (sending/receiving). Assumes email content is provided as input.
*   Integration with live Mercury Insurance quoting APIs. Mock data/logic will be used.
*   Complex multi-turn conversation handling beyond the initial info request/discount check.
*   Advanced error handling or user interface for the human agent.
*   Support for other lines of business (Policy Change, Cancel, etc.).

### Technology
*   **Core Framework:** LangGraph
*   **LLM:** OpenAI GPT-4 Turbo (or similar capable model)
*   **Language:** Python
*   **Libraries:** LangChain

## 2. Architecture Overview

This system utilizes LangGraph to create a stateful agent. The agent's execution flow is defined as a graph where nodes represent processing steps (Python functions) and edges represent transitions based on the current state.

*   **State Graph:** The core LangGraph object managing nodes and edges.
*   **Nodes:** Python functions performing specific tasks like analyzing emails, extracting data, generating requests, or interacting with LLMs/APIs.
*   **State (`AgentState`):** A TypedDict representing the current status of the quoting process, including customer information, missing details, generated messages, etc. The state is passed between nodes and updated at each step.
*   **Edges:** Define the possible transitions between nodes. Conditional edges are used to route the flow based on the data within the `AgentState` (e.g., whether required information is complete).

## 3. Workflow Diagram

```mermaid
graph TD
    A[Customer] -->|Sends Email Inquiry| B(Human Agent/Intake)
    B -->|Passes Email to Agent| C{Identify Intention}
    C -- New Business --> D{Identify Line of Business}
    C -- Other --> Z[End/Manual Handling]
    D -- PPA --> E[Analyze Information (Extract/Check Missing)]
    D -- Other LoB --> Z
    E -- Info Incomplete --> F[Generate Info Request]
    E -- Info Complete --> G[Ask for Discount Proof?]
    F --> H{Agency Review}
    G --> H
    H -- Approved --> I{Send to Customer}
    I -- Customer Responds --> J[Process Customer Response]
    J --> E  // Loop back to analyze new info
    G -- Assume Response Yes/No --> K[Generate Quote (Mock)]
    K --> H
```

**Walkthrough:**

1.  A customer sends an email requesting a quote.
2.  A human agent (or an initial intake system) passes the email content to the AI Agent.
3.  The `identify_intention` node determines if it's a new business request.
4.  If yes, `identify_line_of_business` checks if it's for PPA.
5.  If PPA, `analyze_information` extracts data from the email and compares it against PPA requirements.
6.  If information is missing, `generate_info_request` drafts an email asking for the missing details.
7.  If information is complete, `ask_for_discount_proof` drafts an email asking about discounts.
8.  (Future step after customer response) If info was missing and customer provides it, `process_customer_response` updates the state and loops back to `analyze_information`.
9.  (Future step after customer response) If discount proof info is provided, `process_customer_response` updates the state.
10. If info is complete and discount status known, `generate_quote` creates a mock quote and drafts a quote email.
11. All drafted emails (`generate_info_request`, `ask_for_discount_proof`, `generate_quote`) go to `agency_review`.
12. The human agent reviews the draft. If approved, it's marked for sending (actual sending is out of PoC scope). If rejected/edited, the workflow might require adjustments (TBD).

## 4. State Definition (`AgentState`)

```python
# from typing import TypedDict, List, Dict, Optional

class AgentState(TypedDict):
    customer_email: str       # Raw content of the customer's email(s)
    customer_info: Dict       # Information extracted about the customer/vehicle
    missing_info: List[str]   # List of required fields still missing
    ppa_requirements: List[str] # List of fields needed for a PPA quote
    quote_ready: bool         # Flag indicating if a quote can be generated
    quote_data: Optional[Dict] # Holds the generated (mock) quote data
    proof_of_discount: Optional[bool] # Status of discount proof (True/False/None if unknown)
    # current_action: str     # (Potentially removable if using conditional edges) Tracks last action
    messages: List[Dict]      # History of messages (internal and drafts for customer)
                                # Example message: {'role': 'agent', 'content': '...', 'type': 'info_request', 'requires_review': True}
    requires_review: bool     # Flag indicating if the current state requires human review
```

## 5. Component Details (Nodes)

*   **`identify_intention`**:
    *   Input: `customer_email`
    *   Output: State update (potentially classifying intent), determines next node (new_business or end).
    *   Logic: Uses LLM to classify the primary goal of the email (e.g., seeking new quote vs. policy change).
*   **`identify_line_of_business`**:
    *   Input: `customer_email`
    *   Output: State update (identifying LoB), determines next node (process_ppa or end).
    *   Logic: Uses LLM to determine the insurance type requested (e.g., PPA, Home, etc.).
*   **`analyze_information`**:
    *   Input: `customer_email`, `customer_info`, `ppa_requirements`
    *   Output: Updates `customer_info` with extracted data, updates `missing_info`. Determines next node (`generate_info_request` or `ask_for_discount_proof`).
    *   Logic: Uses LLM to extract structured data based on `ppa_requirements` from the email. Compares extracted data against requirements to find missing fields.
*   **`generate_info_request`**:
    *   Input: `missing_info`, `customer_info`
    *   Output: Appends a draft email message to `messages` list, sets `requires_review` to True.
    *   Logic: Uses LLM to generate a polite email asking the customer for the specific `missing_info`.
*   **`ask_for_discount_proof`**:
    *   Input: `customer_info`
    *   Output: Appends a draft email message to `messages`, sets `requires_review` to True.
    *   Logic: Uses LLM to generate an email asking if the customer has proof for common discounts.
*   **`process_customer_response` (Future Node)**:
    *   Input: New `customer_email` (the reply), current `AgentState`.
    *   Output: Updates `customer_info` and potentially `proof_of_discount` based on the reply. Determines next node (likely back to `analyze_information`).
    *   Logic: Parses the customer's reply, extracts the newly provided information or discount status using LLM.
*   **`generate_quote`**:
    *   Input: `customer_info`, `proof_of_discount` (assumed available and processed)
    *   Output: Updates `quote_data`, `quote_ready`, appends draft quote email to `messages`, sets `requires_review` to True.
    *   Logic: Calls a **mock** quoting function/API based on available info. Uses LLM to generate a quote summary email.
*   **`agency_review`**:
    *   Input: `AgentState` (specifically the latest message in `messages` marked with `requires_review: True`)
    *   Output: State update indicating review outcome (approved/rejected - simplified for PoC). Determines next node (`send_to_customer` or potentially loop back/end).
    *   Logic (PoC): Simulates approval.
    *   Logic (Real): This node would **interrupt** graph execution. An external system/UI would display the draft message to the human agent. The agent's action (Approve/Reject/Edit) would trigger the graph to resume via an external call (e.g., API endpoint associated with the graph execution).

## 6. Human-in-the-Loop (HITL)

The `agency_review` node acts as the HITL checkpoint.
1.  When the workflow reaches a step requiring review (`generate_info_request`, `ask_for_discount_proof`, `generate_quote`), the relevant draft message is added to the state, and the `requires_review` flag is set.
2.  The graph execution should pause (using LangGraph's interruption capabilities).
3.  An external notification mechanism (outside PoC scope) informs the human agent.
4.  The agent accesses the review interface (outside PoC scope), views the draft message and relevant context (`customer_info`, etc.).
5.  The agent provides input (Approve, Reject, potentially Edit - Edit might require more complex state handling).
6.  This input triggers the resumption of the graph execution.
7.  If approved, the workflow proceeds (e.g., to a simulated 'send' state). If rejected, it might end or loop back depending on the design.

For the PoC, this node will likely just simulate approval and proceed.

## 7. LLM Interaction

*   **Model:** GPT-4 Turbo (configurable)
*   **Key Tasks:**
    *   Intent Recognition: Classifying email purpose.
    *   Line of Business Identification: Determining insurance type.
    *   Information Extraction: Pulling structured data (driver name, vehicle details) from unstructured email text.
    *   Text Generation: Drafting emails for information requests, discount queries, and quotes.
*   **Prompts:** Prompts will be designed to be clear, specific, and guide the LLM towards the desired output format (e.g., requesting JSON for extraction). Example (Info Extraction):
    ```
    Extract the following PPA insurance information from the customer email below. Provide the output as a JSON object with keys: {list_of_keys}. If information for a key is not found, use null or an empty string as the value.

    Email:
    "{customer_email_content}"

    JSON Output:
    ```
*   **Parsing:** `JsonOutputParser` from LangChain will be used for LLM calls expected to return JSON. Error handling (e.g., `try-except` blocks) will be added around parsing steps. Consider `PydanticOutputParser` for more robust schema enforcement if needed.

## 8. Data Flow

1.  Initial `customer_email` enters the `AgentState`.
2.  Nodes read relevant parts of the state (`customer_email`, `customer_info`, etc.).
3.  LLM calls process data and generate results (extracted info, draft text).
4.  Node functions update the `AgentState` (`customer_info`, `missing_info`, `messages`, `quote_data`).
5.  Conditional edges read the updated state to determine the next transition.
6.  The state persists and evolves throughout the graph execution.

## 9. Error Handling (Basic)

*   Wrap LLM API calls in `try-except` blocks to catch network/API errors.
*   Wrap JSON parsing in `try-except` blocks to handle malformed LLM outputs.
*   Implement basic checks for unexpected state transitions.
*   Log errors for debugging.
*   For the PoC, unrecoverable errors might simply end the execution.

## 10. Future Considerations / Next Steps

*   **Email Integration:** Implement actual email sending/receiving capabilities.
*   **Real Quoting API:** Replace mock quote generation with calls to Mercury's internal PPA quoting API.
*   **Customer Response Loop:** Fully implement `process_customer_response` and the loop back to `analyze_information`.
*   **HITL Interface:** Develop the UI/system for the human agent review process and the mechanism to resume graph execution.
*   **Robustness:** Enhance error handling, add validation, logging, and monitoring.
*   **State Management:** Consider database persistence for agent state if longer-running interactions are needed.
*   **Scalability:** Evaluate performance and potential for handling concurrent requests.
*   **Expand Scope:** Add support for other lines of business or more complex scenarios within PPA.
*   **Security:** Ensure proper handling of Personally Identifiable Information (PII). 