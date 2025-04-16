# System Design: Granular Human-in-the-Loop (HITL) with Retry Loop for PPA Agent (v2)

**Revision:** 1 (Incorporating Retry Loop)

**Date:** 2025-04-16

## 1. Goal

To enhance the PPA Agent workflow to allow a human insurance agent to review, **correct**, and explicitly approve or reject the AI agent's output at critical intermediate stages (e.g., after information extraction, after discount assessment, before generating a final quote). If rejected due to AI performance issues (e.g., inaccuracies, poor tone, missing details), the workflow should loop back, enabling the AI to retry the specific step, potentially using provided human feedback. This approach provides finer-grained control, ensures human oversight *before* the AI proceeds based on potentially flawed intermediate results, and allows for iterative AI improvement within a single customer interaction.

The system should also clearly represent the distinct roles (Customer, AI Agent, Human Agent) in the visualized conversation flow.

## 2. Proposed High-Level Workflow (with Retry)

The interaction flow becomes more dynamic and collaborative:

1.  **Customer Email Arrival:** A customer email initiates or continues a thread.
2.  **Human Agent Intake (Implicit):** The human agent receives the email in their system (e.g., CRM, inbox).
3.  **Human Agent Invokes AI:** The human agent triggers the PPA AI Agent (LangGraph workflow) with the customer's email content and thread context.
4.  **AI Agent Processing (Segmented):** The AI executes parts of its workflow (e.g., identify intent, analyze info).
5.  **AI Reaches Review Point:** At predefined critical points, the AI prepares its findings or a draft response.
6.  **Workflow Pauses for Review:** The LangGraph workflow *interrupts* execution and waits for human input. The current state (including the data needing review) is persisted.
7.  **Human Agent Review:** A Human Agent UI (external to this agent's core) displays the AI's draft/findings and relevant context. The human agent reviews it.
8.  **Human Agent Decision:** The human agent chooses:
    *   **Accept:** Approves the AI's work for the current step. The workflow resumes and proceeds to the next *distinct* step.
    *   **Reject:** Disapproves of the AI's current output *for that specific step*. The Human Agent UI should ideally allow providing brief feedback/reason for rejection. This feedback is captured.
9.  **Workflow Resumes & Routes:** Based on the human decision:
    *   If **Accepted**, the decision node routes the flow forward to the next logical step.
    *   If **Rejected**, the decision node routes the flow **back to the beginning of the AI step that was just rejected.** The rejection feedback is added to the state.
10. **AI Retries Step (on Rejection):** The AI node executes again. It checks the `AgentState` for rejection feedback from the previous attempt for this specific step and incorporates this feedback into its prompt or logic to generate an improved output. Retry counts are incremented.
11. **Return to Review:** The AI's *new* output for the same step is prepared for review, and the workflow pauses again (Back to step 6/7). This loop continues until the human agent Accepts the output for that step or a retry limit is reached.
12. **Proceed on Acceptance:** Once a step's output is Accepted, the workflow moves forward to the *next* distinct step (e.g., from Info Analysis to Discount Check).
13. **Final Output to Human Agent:** Once the AI completes its overall task (potentially after multiple review cycles), the final proposed response or summary is presented back to the *human agent*.
14. **Human Agent Sends Reply:** The human agent gives the final OK, potentially makes minor final edits, and sends the AI-assisted reply to the customer.

## 3. Key Concepts & Components

*   **Human Agent UI (Conceptual):** An external interface where the human agent manages emails, invokes the AI, sees review requests, provides accept/reject decisions (and feedback), and sends the final email.
*   **LangGraph Workflow:** The state machine defining the AI's tasks, now including specific nodes and edges designed for pausing, resuming, and looping back on rejection.
*   **Interruptible Checkpoints:** Specific edges or nodes in the graph marked as interruptible using LangGraph's capabilities. When the workflow hits these points, execution pauses.
*   **Persistence Layer:** Required to save the graph's state when it pauses, allowing it to be resumed later (LangGraph offers various backends like SQLite, Postgres, etc.).
*   **Review Nodes:** Nodes dedicated to preparing data for human review (`Prepare...Review`) and nodes to handle the routing based on the human decision (`Process...Decision`).
*   **`AgentState` Modifications:** The state needs refinement to track the granular review process (See Section 5).
*   **API/Callback Mechanism (Conceptual):** A way for the Human Agent UI to send the review decision and feedback back to resume the specific paused LangGraph execution thread.
*   **Retry Limits:** Mechanism to prevent infinite loops if the AI repeatedly fails a step (e.g., max 2-3 retries per step).

## 4. Detailed LangGraph Workflow Structure (Conceptual)

This diagram illustrates the review loop concept for the 'Analyze Information' step. Similar loops would exist for other review points (Discount Check, Quote Generation, etc.).

```mermaid
graph TD
    Start[Customer Email Received] --> HumanIntake{Human Agent Views Email};
    HumanIntake -- Invoke AI --> WF_Start(Workflow Start);

    WF_Start --> IdentifyIntent(Identify Intent);
    IdentifyIntent --> AnalyzeInfo(Analyze Information);
    AnalyzeInfo --> PrepareInfoReview(Prepare Info for Review);
    PrepareInfoReview --> Pause_InfoReview{Wait for Human Review (Info)};

    Pause_InfoReview -- Human Input --> ProcessInfoDecision{Process Info Review Decision};

    ProcessInfoDecision -- Accepted --> CheckDiscounts(Check for Discounts); %% --> Proceed to next distinct step
    ProcessInfoDecision -- Rejected & Retries OK --> AnalyzeInfo; %% <-- Loop back to retry the *same* step
    ProcessInfoDecision -- Rejected & Max Retries --> HandleMaxRetries(Handle Max Retries / Error);

    CheckDiscounts --> PrepareDiscountReview(Prepare Discount Info/Query for Review);
    PrepareDiscountReview --> Pause_DiscountReview{Wait for Human Review (Discount)};
    Pause_DiscountReview -- Human Input --> ProcessDiscountDecision{Process Discount Review Decision};
    ProcessDiscountDecision -- Accepted --> CheckQuoteReady(Quote Ready?);
    ProcessDiscountDecision -- Rejected & Retries OK --> CheckDiscounts;
    ProcessDiscountDecision -- Rejected & Max Retries --> HandleMaxRetries;

    CheckQuoteReady -- Yes --> GenerateQuote(Generate Quote);
    GenerateQuote --> PrepareQuoteReview(Prepare Quote for Review);
    PrepareQuoteReview --> Pause_QuoteReview{Wait for Human Review (Quote)};
    Pause_QuoteReview -- Human Input --> ProcessQuoteDecision{Process Quote Review Decision};
    ProcessQuoteDecision -- Accepted --> FormatFinalResponse(Format Final Response for Human);
    ProcessQuoteDecision -- Rejected & Retries OK --> GenerateQuote;
    ProcessQuoteDecision -- Rejected & Max Retries --> HandleMaxRetries;

    CheckQuoteReady -- No --> FormatInfoRequest(Format Info Request for Human); 
    FormatInfoRequest --> PrepareInfoRequestReview(Prepare Info Request for Review); %% Optional: Review info requests too
    PrepareInfoRequestReview --> Pause_InfoRequestReview{Wait for Human Review (Info Request)};
    Pause_InfoRequestReview -- Human Input --> ProcessInfoRequestDecision{Process Info Request Decision};
    ProcessInfoRequestDecision -- Accepted --> FormatFinalResponse;
    ProcessInfoRequestDecision -- Rejected & Retries OK --> FormatInfoRequest;
    ProcessInfoRequestDecision -- Rejected & Max Retries --> HandleMaxRetries;

    FormatFinalResponse --> WF_End(Workflow End - Output to Human);
    WF_End --> HumanFinalReview{Human Agent Reviews Final AI Output};
    HumanFinalReview -- Send --> SendToCustomer[Send Email to Customer];

    HandleMaxRetries --> WF_End_Error(Workflow End - Error);

    style Pause_InfoReview fill:#ff9,stroke:#333,stroke-width:2px
    style Pause_DiscountReview fill:#ff9,stroke:#333,stroke-width:2px
    style Pause_QuoteReview fill:#ff9,stroke:#333,stroke-width:2px
    style Pause_InfoRequestReview fill:#ff9,stroke:#333,stroke-width:2px
    style HandleMaxRetries fill:#f99,stroke:#333,stroke-width:1px
    style WF_End_Error fill:#f66,stroke:#333,stroke-width:2px
```

*   **Yellow Nodes:** Represent points where the workflow pauses, waiting for the human agent.
*   **Decision Nodes:** Route based on 'Accepted' vs 'Rejected'.
*   **Retry Loops:** Edges routing back from 'Rejected' to the beginning of the same logical step.
*   **Retry Limit Handling:** Explicit path for handling exceeded retries.

## 5. `AgentState` Modifications (`src/ppa_agent/state.py`)

The `AgentState` TypedDict needs updates:

*   **`messages`:** This list should store the *final, human-approved* messages intended for the customer, or potentially internal AI-to-AI messages if needed. Drafts for review should be handled differently.
*   **`email_thread`:** This should accurately reflect the multi-party conversation: Customer emails, Human Agent actions/forwards (optional detail), AI drafts presented for review, Human review actions (Accept/Reject/Feedback), and the final Human Agent reply based on the approved AI draft.
    *   Consider adding a `source` field to each entry: `'customer'`, `'ai_draft'`, `'human_review_action'`, `'human_final_message'`. 
*   **`step_requiring_review`:** (Optional[str]) - Name of the step currently paused for review (e.g., `'info_analysis'`, `'discount_check'`). Null if not paused.
*   **`data_for_review`:** (Optional[Dict]) - The specific data payload generated by the AI that needs human review (e.g., the extracted info dictionary, the proposed discount query text).
*   **`last_human_decision_for_step`:** (Optional[Dict[str, str]]) - Stores the decision ('accepted'/'rejected') *per reviewable step*. Example: `{'info_analysis': 'rejected', 'discount_check': 'accepted'}`. This should be cleared or updated when a step is successfully accepted.
*   **`rejection_feedback_for_step`:** (Optional[Dict[str, str]]) - Stores human feedback *per step* when rejected. Example: `{'info_analysis': 'Missed the vehicle year mention on line 3'}`. Should be consumed and cleared by the AI node upon retry.
*   **`retry_counts`:** (Optional[Dict[str, int]]) - Tracks retries per step. Example: `{'info_analysis': 1}`. Used to prevent infinite loops.

## 6. Visualization Changes (`src/visualize_agent.py`)

*   **Conversation History:** Modify generation logic (`generate_conversation_html` or similar):
    *   Iterate through the enhanced `email_thread` (or a similar structured history).
    *   Use distinct styling (CSS classes) for:
        *   `customer-turn`
        *   `ai-draft-turn` (AI output presented for review)
        *   `human-review-turn` (Shows 'Agent reviewed: Accepted' or 'Agent reviewed: Rejected. Feedback: ...')
        *   `human-final-turn` (The message approved/edited by the human agent for sending)
*   **State Display:** Clearly show the new `AgentState` fields related to review status, feedback, and retries in the final state JSON/HTML display.
*   **Graph Visualization:** While the Mermaid diagram shows the static structure, the HTML report should ideally simulate the flow dynamically or provide snapshots:
    *   Snapshot 1: Before review pause.
    *   Snapshot 2: After rejection and feedback added (State update).
    *   Snapshot 3: After successful retry and acceptance.

## 7. Implementation Considerations (PoC Scope)

*   **Simulated Pauses & Resumption:** For `visualize_agent.py`, we won't implement true asynchronous pausing/resuming with persistence.
    *   We'll run the graph until it *would* pause for review.
    *   Simulate human **rejection**: Manually update the state with the decision ('rejected'), feedback, and increment retry count.
    *   *(Challenge)* Attempt to invoke the *specific node* or *resume the graph* with the updated state to simulate the retry. If LangGraph's synchronous `invoke` makes resumption difficult, we might need to show the state *before* retry and the state *after* a simulated successful retry as separate steps.
    *   Simulate human **acceptance**: Manually update the state with 'accepted' and invoke the next part of the workflow.
*   **Persistence Layer:** Not implemented for the visualization script.
*   **Human Agent UI/API:** Completely outside the scope; we only simulate the *effect* of the human decision on the `AgentState`.
*   **Retry Limits:** Implement checks within the decision nodes (`Process...Decision`) based on `retry_counts`.
*   **Feedback Incorporation:** AI nodes (`AnalyzeInfo`, `CheckDiscounts`, etc.) need modification to check for and incorporate rejection feedback from the state into their prompts/logic.

## 8. Benefits

*   **Increased Accuracy:** Human oversight corrects potential AI errors early and iteratively.
*   **Enhanced Trust & Control:** Human agents remain firmly in control of the process and outputs.
*   **Better Audit Trail:** Clear record of AI suggestions, human decisions, and feedback loops.
*   **Handles Ambiguity:** Humans can resolve situations the AI finds unclear.
*   **Iterative AI Improvement:** Allows the AI to learn from human correction *within the same task*, reducing repetitive errors.
*   **Reduces Human Effort:** Shifts effort from manual correction to guided AI self-correction.
