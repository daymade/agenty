from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ppa_agent.agent import PPAAgent

app = FastAPI(title="PPA New Business AI Agent API")
agent = PPAAgent()

class StartRequest(BaseModel):
    email: str

class ReviewRequest(BaseModel):
    thread_id: str
    decision: str
    feedback: Optional[str] = None

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/start")
async def start(req: StartRequest) -> Dict[str, Any]:
    """Kick off a new conversation by processing customer email."""
    try:
        state = agent.process_email(email_content=req.email, thread_id=None)
        return {"thread_id": state.get("thread_id"), "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/review")
async def review(req: ReviewRequest) -> Dict[str, Any]:
    """Resume workflow after human review decision."""
    try:
        state = agent.resume_after_review(thread_id=req.thread_id, decision=req.decision, feedback=req.feedback)
        return {"thread_id": req.thread_id, "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
