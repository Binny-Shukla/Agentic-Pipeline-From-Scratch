# integrated_system/task_bridge.py
import json
from typing import Any, Dict, List


class TaskBridge:
    """
    Bridges communication between:
    - Seq2Seq Task Representation (JSON)
    - Reasoning Model (Symbolic Token IDs & Action Trajectories)
    - Agentic Decoder (ChatML System Prompt & Context)
    """

    def __init__(self):
        pass

    def task_to_reasoning_ids(self, task_dict: Dict[str, Any]) -> List[int]:
        """
        Converts parsed JSON task into an integer token ID sequence (0..199)
        for the Reasoning Model encoder.
        """
        # Create a deterministic character-hash sequence or token representation
        serialized = json.dumps(task_dict, sort_keys=True)
        # Map characters to bounded token IDs within the 200-token symbolic vocab
        token_ids = [(ord(char) % 190) + 5 for char in serialized[:32]]
        if not token_ids:
            token_ids = [2, 10, 20, 30, 4]
        return token_ids

    def build_decoder_prompt(
        self,
        user_prompt: str,
        task_dict: Dict[str, Any],
        semantic_actions: List[str],
        candidate_tools: List[str],
        verifier_verdict: str = "ACCEPT",
    ) -> str:
        """
        Formats a complete ChatML prompt for the Agentic Decoder,
        embedding the plan, verifier status, and available tools.
        """
        domain = task_dict.get("domain", "general")
        objective = task_dict.get("objective", user_prompt)
        plan_str = " -> ".join(semantic_actions) if semantic_actions else "execute"
        tools_str = ", ".join(candidate_tools) if candidate_tools else "python, calculator"

        prompt = (
            f"<|im_start|>system\n"
            f"You are Cherry, an autonomous agentic AI with reasoning, tool execution, and self-repair capabilities.\n"
            f"Domain: {domain}\n"
            f"Objective: {objective}\n"
            f"Tactical Plan: [{plan_str}]\n"
            f"Verification Baseline: {verifier_verdict}\n"
            f"Candidate Tools: [{tools_str}]\n"
            f"Follow this protocol:\n"
            f"1. Think first inside <|analysis|> reasoning steps <|analysis_end|>.\n"
            f"2. If external information or computation is needed, invoke a tool using:\n"
            f"<|tool_call|>\n"
            f"{{\n"
            f'  "tool": "<tool_name>",\n'
            f'  "args": {{ ... }}\n'
            f"}}\n"
            f"<|tool_end|>\n"
            f"3. After tool observation returns, continue to formulate your verified answer.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_prompt}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt
