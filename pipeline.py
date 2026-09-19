# integrated_system/pipeline.py
import re
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from integrated_system.models.seq2seq import Seq2SeqPipeline
from integrated_system.models.reasoning import ReasoningPipeline
from integrated_system.models.decoder import AgenticDecoderPipeline
from integrated_system.task_bridge import TaskBridge

# Add tools directory to path
TOOLS_DIR = Path(__file__).resolve().parent.parent / "Self-Perpetuating-Model" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from action_adaptor import ActionAdapter
from tool_router import ToolRouter
from tool_registry import ToolRegistry
from tool_executor import ToolExecutor, ToolResult


class CherryAgent:
    """
    Unified Cherry Autonomous Agent.
    Orchestrates:
    - Seq2Seq Task Parsing
    - Reasoning Planning, Best-First Search & Multi-Stage Verification
    - Action Adaptation & Tool Routing
    - Agentic Decoder with 32-slot Neural Working Memory
    - Safe Sandboxed Tool Execution & Autonomous Self-Repair
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[CherryAgent] Initializing agent on device: {self.device}")

        # 1. Models
        print("[CherryAgent] Loading Seq2Seq model...")
        self.seq2seq = Seq2SeqPipeline(device=self.device)

        print("[CherryAgent] Loading Reasoning model...")
        self.reasoning = ReasoningPipeline(device=self.device)

        print("[CherryAgent] Loading Agentic Decoder with Neural Working Memory...")
        self.decoder = AgenticDecoderPipeline(device=self.device)

        # 2. Tools & Routing
        self.task_bridge = TaskBridge()
        self.action_adapter = ActionAdapter()
        self.tool_router = ToolRouter()
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_default_tools()
        self.tool_executor = ToolExecutor(self.tool_registry)

        print("[CherryAgent] Initialization complete. Available tools:", self.tool_registry.list_tools())

    def parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts tool name and arguments from <|tool_call|> ... <|tool_end|> block.
        """
        match = re.search(r"<\|tool_call\|>(.*?)(?:<\|tool_end\|>|$)", text, re.DOTALL)
        if not match:
            return None

        raw_payload = match.group(1).strip()
        # Clean potential markdown formatting
        if raw_payload.startswith("```json"):
            raw_payload = raw_payload[7:]
        if raw_payload.startswith("```"):
            raw_payload = raw_payload[3:]
        if raw_payload.endswith("```"):
            raw_payload = raw_payload[:-3]
        raw_payload = raw_payload.strip()

        try:
            parsed = json.loads(raw_payload)
            if isinstance(parsed, dict) and "tool" in parsed:
                return parsed
        except Exception:
            pass

        # Regex fallback for {"tool": "...", "args": ...}
        tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', raw_payload)
        if tool_match:
            tool_name = tool_match.group(1)
            return {"tool": tool_name, "args": {}}

        return None

    def run(
        self,
        user_prompt: str,
        max_tool_turns: int = 5,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Full autonomous end-to-end execution loop.
        """
        history_trace: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # STEP 1: PARSING (Seq2Seq Transformer)
        # -------------------------------------------------------------
        task_dict = self.seq2seq.parse_task(user_prompt)
        history_trace.append({"stage": "seq2seq_parsing", "output": task_dict})

        # -------------------------------------------------------------
        # STEP 2: STRATEGIC PLANNING & SEARCH (Reasoning Model)
        # -------------------------------------------------------------
        reasoning_ids = self.task_bridge.task_to_reasoning_ids(task_dict)
        plan_output = self.reasoning.plan_task(reasoning_ids)
        searched_action_ids = plan_output["searched_actions"]
        verification = plan_output["verification"]

        history_trace.append({
            "stage": "reasoning_planning",
            "searched_action_ids": searched_action_ids,
            "verification": verification,
        })

        # -------------------------------------------------------------
        # STEP 3: ACTION ADAPTATION & TOOL ROUTING
        # -------------------------------------------------------------
        adapted_actions = self.action_adapter.adapt(searched_action_ids)
        action_names = [cand.action for cand in adapted_actions]

        candidate_tools: List[str] = []
        for action in action_names:
            try:
                route = self.tool_router.route(action)
                for tool in route.tools:
                    if tool not in candidate_tools and self.tool_registry.has(tool):
                        candidate_tools.append(tool)
            except ValueError:
                pass

        if not candidate_tools:
            candidate_tools = ["calculator", "python", "file_read"]

        history_trace.append({
            "stage": "tool_routing",
            "semantic_actions": action_names,
            "candidate_tools": candidate_tools,
        })

        # -------------------------------------------------------------
        # STEP 4: GENERATION WITH DECODER & NEURAL WORKING MEMORY
        # -------------------------------------------------------------
        current_context = self.task_bridge.build_decoder_prompt(
            user_prompt=user_prompt,
            task_dict=task_dict,
            semantic_actions=action_names,
            candidate_tools=candidate_tools,
            verifier_verdict=verification["decision_label"],
        )

        memory_state = None  # 32-slot Neural Working Memory state
        tool_call_records: List[Dict[str, Any]] = []

        for turn_idx in range(max_tool_turns):
            # Generate next segment
            generated_chunk, _, memory_state = self.decoder.generate(
                prompt_text=current_context,
                memory_state=memory_state,
                max_new_tokens=256,
                temperature=temperature,
                stop_on_tool_call=True,
            )

            current_context += generated_chunk

            # Check if generation emitted a tool call
            tool_call_data = self.parse_tool_call(generated_chunk)

            if not tool_call_data:
                # No more tool calls; agent has completed its response
                break

            # Execute tool call
            tool_name = tool_call_data.get("tool", "")
            tool_args = tool_call_data.get("args", {})

            if not isinstance(tool_args, dict):
                tool_args = {}

            execution_result = self.tool_executor.execute(tool_name, tool_args)
            tool_record = {
                "turn": turn_idx + 1,
                "tool": tool_name,
                "args": tool_args,
                "success": execution_result.success,
                "output": execution_result.output,
                "error": execution_result.error,
                "execution_time": execution_result.execution_time,
            }
            tool_call_records.append(tool_record)

            # Step 5: Inject tool observation back into context
            obs_payload = {
                "success": execution_result.success,
                "tool": tool_name,
                "output": str(execution_result.output) if execution_result.output is not None else None,
                "error": execution_result.error,
            }

            obs_turn = (
                f"\n<|im_start|>tool\n"
                f"{json.dumps(obs_payload)}\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            # If tool execution failed, inject self-repair reasoning
            if not execution_result.success:
                obs_turn += f"<|analysis|>Tool execution returned an error: {execution_result.error}. Analyzing and self-repairing.<|analysis_end|>\n"

            current_context += obs_turn

        # Extract final assistant response
        # Find the last <|im_start|>assistant segment
        assistant_segments = current_context.split("<|im_start|>assistant\n")
        raw_final = assistant_segments[-1] if len(assistant_segments) > 1 else current_context
        # Strip trailing <|im_end|>
        final_answer = raw_final.replace("<|im_end|>", "").strip()

        return {
            "prompt": user_prompt,
            "task_analysis": task_dict,
            "tactical_plan": action_names,
            "verifier_verdict": verification["decision_label"],
            "tool_calls": tool_call_records,
            "final_response": final_answer,
            "full_conversation": current_context,
            "trace": history_trace,
        }
