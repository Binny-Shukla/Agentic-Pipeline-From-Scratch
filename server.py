# integrated_system/server.py
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from integrated_system.pipeline import CherryAgent

# Initialize FastAPI App
app = FastAPI(
    title="Cherry AI Integrated Service",
    description="Autonomous Agentic AI API uniting Seq2Seq Task Parsing, Strategic Reasoning & Planning, 32-Slot Neural Memory Decoder, and Sandboxed Tools.",
    version="1.0.0",
)

# Enable CORS for cross-project and web frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Agent Instance (Lazy/Startup loaded)
agent: Optional[CherryAgent] = None


@app.on_event("startup")
def startup_event():
    global agent
    if agent is None:
        print("[Server] Initializing CherryAgent models on startup...")
        agent = CherryAgent()
        print("[Server] CherryAgent models loaded and ready.")


def get_agent() -> CherryAgent:
    global agent
    if agent is None:
        agent = CherryAgent()
    return agent


# =============================================================================
# Request / Response Schemas
# =============================================================================

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User query or task prompt", example="Calculate 25 * 40 - 150")
    max_tool_turns: int = Field(default=5, ge=1, le=15, description="Maximum autonomous tool feedback turns")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature for Decoder")


class ToolCallRecord(BaseModel):
    turn: int
    tool: str
    args: Dict[str, Any]
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time: float


class ChatResponse(BaseModel):
    prompt: str
    final_response: str
    task_analysis: Dict[str, Any]
    tactical_plan: List[str]
    verifier_verdict: str
    tool_calls: List[ToolCallRecord]
    latency_seconds: float


class ParseTaskRequest(BaseModel):
    prompt: str = Field(..., description="Natural language task prompt to normalize", example="Train a churn model")


class PlanTaskRequest(BaseModel):
    task_token_ids: Optional[List[int]] = Field(default=None, description="Optional symbolic token IDs")


class ExecuteToolRequest(BaseModel):
    tool: str = Field(..., description="Tool name (calculator, python, file_read, etc.)", example="calculator")
    args: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the tool", example={"expression": "12 * 12"})


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    """
    Returns server status, compute device, and available tools.
    """
    curr_agent = get_agent()
    return {
        "status": "healthy",
        "service": "Cherry AI Integrated System",
        "device": str(curr_agent.device),
        "models": {
            "seq2seq": "Loaded (full_encoder_decoder.pt, 505K params, vocab=467)",
            "reasoning": "Loaded (best_reason_model.pt, state_dim=256, action_dim=10)",
            "decoder": "Loaded (best_decoder_mem.pt, 14.4M params, 32-slot Neural Memory)",
        },
        "available_tools": curr_agent.tool_registry.list_tools(),
    }


@app.post("/chat", response_model=ChatResponse, tags=["Autonomous Agent"])
def chat(request: ChatRequest):
    """
    End-to-End Autonomous Execution:
    Prompt -> Seq2Seq Normalization -> Reasoning Planning -> Decoder Generation <-> Tool Execution -> Final Verified Answer.
    """
    curr_agent = get_agent()
    t0 = time.perf_counter()

    try:
        result = curr_agent.run(
            user_prompt=request.prompt,
            max_tool_turns=request.max_tool_turns,
            temperature=request.temperature,
        )
        latency = time.perf_counter() - t0

        return ChatResponse(
            prompt=result["prompt"],
            final_response=result["final_response"],
            task_analysis=result["task_analysis"],
            tactical_plan=result["tactical_plan"],
            verifier_verdict=result["verifier_verdict"],
            tool_calls=result["tool_calls"],
            latency_seconds=round(latency, 3),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution error: {str(e)}",
        )


@app.post("/parse_task", tags=["Component Access"])
def parse_task(request: ParseTaskRequest):
    """
    Directly query the Seq2Seq Transformer to convert natural language into a canonical task dictionary.
    """
    curr_agent = get_agent()
    try:
        parsed = curr_agent.seq2seq.parse_task(request.prompt)
        return {"prompt": request.prompt, "canonical_task": parsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan_task", tags=["Component Access"])
def plan_task(request: PlanTaskRequest):
    """
    Directly query the Reasoning Model to produce tactical plan steps, Best-First Search trajectories, and verification verdicts.
    """
    curr_agent = get_agent()
    try:
        plan_res = curr_agent.reasoning.plan_task(request.task_token_ids)
        adapted = curr_agent.action_adapter.adapt(plan_res["searched_actions"])
        semantic_actions = [a.action for a in adapted]
        return {
            "planned_action_ids": plan_res["planned_actions"],
            "searched_action_ids": plan_res["searched_actions"],
            "semantic_actions": semantic_actions,
            "verification": plan_res["verification"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute_tool", tags=["Tools"])
def execute_tool(request: ExecuteToolRequest):
    """
    Execute any registered tool directly in the sandboxed environment.
    """
    curr_agent = get_agent()
    if not curr_agent.tool_registry.has(request.tool):
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{request.tool}' not found. Available: {curr_agent.tool_registry.list_tools()}",
        )

    res = curr_agent.tool_executor.execute(request.tool, request.args)
    return {
        "tool": request.tool,
        "args": request.args,
        "success": res.success,
        "output": res.output,
        "error": res.error,
        "execution_time": res.execution_time,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("integrated_system.server:app", host="0.0.0.0", port=8000, reload=False)
