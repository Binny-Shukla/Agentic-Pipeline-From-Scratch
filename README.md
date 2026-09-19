# Cherry: Self-Perpetuating Autonomous AI System

Cherry is a from-scratch, self-perpetuating AI agentic architecture designed to reason, plan, execute tools in sandboxed environments, verify outcomes, detect errors, and autonomously heal itself.

The project unites four specialized neural and algorithmic tiers into a cohesive closed-loop system:
1. **Natural Language Task Normalizer** (Seq2Seq Transformer)
2. **Strategic Reasoning & Planning Engine** (Planner + Best-First Search + Multi-Stage Verifier)
3. **Agentic Generative Decoder** (Differentiable 32-slot Neural Working Memory + ChatML)
4. **Autonomous Tools Suite & Sandbox** (Action Adapter, Tool Router, Tool Registry, Subprocess Executor)

---

## Architecture Overview

```
                          ┌──────────────────────────┐
                          │    User Prompt / Query   │
                          └─────────────┬────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  1. TASK NORMALIZATION & PARSING (Seq2Seq Transformer)                 │
    │  - Checkpoint: full_encoder_decoder.pt (505K params, vocab=467)       │
    │  - Transforms arbitrary natural language into canonical JSON task      │
    │    structures (domain, task, objective, inputs, outputs, constraints). │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  2. STRATEGIC REASONING & PLANNING (Reasoning Model)                   │
    │  - Checkpoint: best_reason_model.pt (~10 MB, action_dim=10)           │
    │  - 5-step learnable tactical planner (PlanIt)                          │
    │  - Best-First Search (BFS) over candidate action sequences             │
    │  - Multi-Stage Verifier (Triple-gate check: Goal, State, Safety)       │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  3. ACTION ADAPTATION & TOOL ROUTING                                   │
    │  - ActionAdapter maps action IDs (0..9) -> Semantic actions            │
    │    (inspect, prepare, build, execute, test, repair, observe, verify)   │
    │  - ToolRouter maps actions -> Candidate tools (python, shell, calc...) │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  4. GENERATION WITH NEURAL WORKING MEMORY (Agentic Decoder)            │
    │  - Checkpoint: best_decoder_mem.pt (14.4M params, d_model=384)        │
    │  - Custom 16,000-token BPE tokenizer (cherry_tokenizer)                │
    │  - 32-slot Slot-Addressed Differentiable Working Memory                │
    │  - ChatML Token Protocol:                                              │
    │      <|analysis|>    -> Internal Chain-of-Thought reasoning            │
    │      <|tool_call|>   -> Structured tool invocation JSON                │
    │      <|im_start|>tool-> Tool observation feedback turn                 │
    └───────────────────────┬────────────────────────▲───────────────────────┘
                            │                        │
               Dispatches   │                        │ Returns Observation
               Tool Call    ▼                        │ & Updates Memory
    ┌────────────────────────────────────────────────┴───────────────────────┐
    │  5. AUTONOMOUS TOOL RUNTIME (ToolExecutor & Sandboxed Environment)     │
    │  - Tools: calculator, python (subprocess), file_read, file_write,      │
    │           file_list, file_search, pip_install, shell, run_tests        │
    │  - If an error occurs: triggers autonomous self-repair trajectory      │
    └────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components Specification

### 1. Seq2Seq Task Parser
- **Checkpoint**: `Self-Perpetuating-Model/Seq_to_Seq/full_encoder_decoder.pt` (~2.3 MB)
- **Architecture**: 4 encoder layers, 4 decoder layers, $d_{\text{model}}=64$, 4 heads, GELU FFN ($d_{\text{ff}}=256$).
- **Vocabulary**: 467 canonical tokens.
- **Function**: Converts messy user instructions into normalized, machine-readable task objects with zero hallucinated requirements.

### 2. Reasoning & Planning Engine
- **Checkpoint**: `Self-Perpetuating-Model/Reasoning model/best_reason_model.pt` (~10 MB)
- **Architecture**: Transformer encoder + 32-slot Neural Memory (`state_dim=256`) + PlanIt (5 plan steps) + SearchIt + MultiStageVerifier.
- **Actions (10)**: `inspect`, `prepare`, `build`, `execute`, `test`, `repair`, `observe`, `verify`, `export`, `no_op`.
- **Search Algorithm**: Best-First Search prioritizing by path cost + neural heuristic $f(n) = g(n) + h(n)$ across 32 state expansions.
- **Triple-Gate Verification**:
  - Head 1: Goal completion check
  - Head 2: State consistency check
  - Head 3: Safety constraint check
  - Emits: `ACCEPT` (proceed), `REPLAN` (divert to self-repair), or `REJECT` (refuse unsafe ops).

### 3. Agentic Decoder with Neural Working Memory
- **Checkpoint**: `Self-Perpetuating-Model/Decoder/Agentic_AI/best_decoder_mem.pt` (~55 MB, 14.4M parameters)
- **Architecture**: 4 layers, 6 heads, $d_{\text{model}}=384$, $d_{\text{ff}}=1536$.
- **Neural Working Memory**: Maintains 32 dynamic memory slots ($32 \times 384$). Cross-attention queries historical context; slot-wise adaptive gating updates memory with observations across chunks without blowing up the 512 context window.
- **Tokenizer**: 16,000 BPE vocabulary with 8 dedicated control tokens:
  `<|pad|>`, `<|unk|>`, `<|im_start|>`, `<|im_end|>`, `<|analysis|>`, `<|analysis_end|>`, `<|tool_call|>`, `<|tool_end|>`.

### 4. Tools Framework & Execution Sandbox
- **Location**: `Self-Perpetuating-Model/tools/`
- **Modules**:
  - `ActionAdapter`: Bridges numeric action IDs into semantic operations.
  - `ToolRouter`: Maps semantic operations to ordered candidate tools.
  - `ToolRegistry`: Manages instances of all 9 system tools.
  - `ToolExecutor`: Safely runs tool executions with timeouts (default 30s) and structured outputs.
- **Available Tools**:
  - `calculator`: Safe mathematical expression parsing via Abstract Syntax Trees.
  - `python`: Controlled subprocess Python code execution.
  - `file_read`, `file_write`, `file_list`, `file_search`: Sandboxed file system operations.
  - `shell`: Subprocess system commands.
  - `pip_install`: Dynamic dependency installation.
  - `run_tests`: Automated test runner.

---

## Integrated Capabilities

1. **Natural Language Understanding**: Maps arbitrary requests to formal specifications.
2. **Transparent CoT Reasoning**: Thinks privately inside `<|analysis|>` tags before generating answers.
3. **Proactive Multi-Step Planning**: Formulates a 5-step tactical roadmap before touching tools.
4. **Heuristic Action Search**: Searches graph state transitions using neural cost estimates.
5. **Structured Tool Calling**: Generates parse-safe JSON tool calls directly within generation stream.
6. **Persistent Working Memory**: 32 neural memory slots track progress and observations across turns.
7. **Autonomous Error Recovery**: Catches runtime errors (tracebacks, syntax errors) and initiates repair sub-plans.
8. **Triple-Gate Safety**: Rejects destructive operations and verifies state transitions.
9. **Calibrated Refusal**: Proposes safe alternatives when commands violate safety gates.

---

## Directory Layout

```
Dominance/
├── integrated_system/              # Unified production package
│   ├── __init__.py                 # Exports CherryAgent
│   ├── server.py                   # FastAPI REST API application
│   ├── pipeline.py                 # Master CherryAgent closed-loop orchestrator
│   ├── task_bridge.py              # Bridges symbolic & 16K BPE representations
│   ├── demo.py                     # 3 live demonstration examples
│   ├── verify_integration.py       # Comprehensive 5-stage automated test suite
│   ├── test_server.py              # FastAPI endpoint test suite
│   └── models/
│       ├── __init__.py             # Exports individual pipelines
│       ├── seq2seq.py              # Standalone Seq2Seq Transformer model
│       ├── reasoning.py            # Standalone Reasoning model + BFS + Verifier
│       └── decoder.py              # Standalone Agentic Decoder + Neural Memory
│
├── Self-Perpetuating-Model/        # Original research models & weights
│   ├── Decoder/Agentic_AI/         # best_decoder_mem.pt (14.4M)
│   ├── Reasoning model/            # best_reason_model.pt (~10 MB)
│   ├── Seq_to_Seq/                 # full_encoder_decoder.pt (~2.3 MB)
│   └── tools/                      # Tool framework (registry, executor, router)
│
├── cherry_tokenizer/               # 16,000 BPE Tokenizer
└── AI_CAN_WORK_HERE/               # Architecture inspection and integration specs
```

---

## Getting Started & Usage

### 1. Requirements

Ensure PyTorch, Transformers, FastAPI, and Uvicorn are installed:

```bash
pip install torch transformers fastapi uvicorn pydantic
```

### 2. Python SDK Usage

You can use the unified `CherryAgent` directly in Python:

```python
from integrated_system import CherryAgent

# Automatically selects CUDA if available, else CPU
agent = CherryAgent()

# Run an autonomous query
result = agent.run("Evaluate the expression: 144 / 12 + (8 * 9) - 15")

print("Domain:", result["task_analysis"]["domain"])
print("Plan:", result["tactical_plan"])
print("Verifier:", result["verifier_verdict"])
print("Tool Calls:", result["tool_calls"])
print("Answer:\n", result["final_response"])
```

### 3. Run Live Demonstration

To see the system execute math calculations, sandboxed Python code, and file inspections:

```bash
python integrated_system/demo.py
```

### 4. Run Automated Verification Suite

To verify all components and test weights loading:

```bash
python integrated_system/verify_integration.py
```

---

## FastAPI REST API

The system includes a production-ready FastAPI server with full CORS support for external apps, web frontends, and VS Code extensions.

### Starting the Server

```bash
python -m uvicorn integrated_system.server:app --host 0.0.0.0 --port 8000
```

- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **ReDoc UI**: `http://localhost:8000/redoc`

### API Endpoints

#### `GET /health`
Returns system status, active compute device, loaded models, and registered tools.

**Response**:
```json
{
  "status": "healthy",
  "service": "Cherry AI Integrated System",
  "device": "cuda",
  "models": {
    "seq2seq": "Loaded (full_encoder_decoder.pt, 505K params, vocab=467)",
    "reasoning": "Loaded (best_reason_model.pt, state_dim=256, action_dim=10)",
    "decoder": "Loaded (best_decoder_mem.pt, 14.4M params, 32-slot Neural Memory)"
  },
  "available_tools": [
    "file_read", "file_list", "file_search", "file_write",
    "calculator", "python", "pip_install", "shell", "run_tests"
  ]
}
```

#### `POST /chat`
Runs the complete autonomous loop and returns the verified response along with the execution trace.

**Request Body**:
```json
{
  "prompt": "Calculate (45 * 2) + 10 and verify the result",
  "max_tool_turns": 3,
  "temperature": 0.7
}
```

**Response**:
```json
{
  "prompt": "Calculate (45 * 2) + 10 and verify the result",
  "final_response": "100",
  "task_analysis": {
    "domain": "data processing",
    "task": "cleaning",
    "objective": "calculate arithmetic expression"
  },
  "tactical_plan": ["inspect", "build", "execute", "verify"],
  "verifier_verdict": "ACCEPT",
  "tool_calls": [
    {
      "turn": 1,
      "tool": "calculator",
      "args": { "expression": "(45 * 2) + 10" },
      "success": true,
      "output": 100,
      "execution_time": 0.0001
    }
  ],
  "latency_seconds": 1.28
}
```

#### `POST /execute_tool`
Directly calls any sandboxed tool.

**Request Body**:
```json
{
  "tool": "calculator",
  "args": { "expression": "2 ** 10 + 24" }
}
```

**Response**:
```json
{
  "tool": "calculator",
  "args": { "expression": "2 ** 10 + 24" },
  "success": true,
  "output": 1048,
  "error": null,
  "execution_time": 0.0001
}
```

#### `POST /parse_task`
Converts natural language into canonical task JSON using the Seq2Seq Transformer.

#### `POST /plan_task`
Queries the Reasoning Model for action trajectories and verification states.

---

## License

GNU
