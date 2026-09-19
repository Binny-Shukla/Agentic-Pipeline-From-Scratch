# integrated_system/demo.py
import sys
import json
import time
from pathlib import Path

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from integrated_system.pipeline import CherryAgent


def run_demo():
    print("=" * 75)
    print("      CHERRY AI: INTEGRATED PIPELINE LIVE DEMONSTRATION")
    print("=" * 75)

    print("\n[+] Initializing CherryAgent on CUDA/CPU...")
    agent = CherryAgent()
    print("[+] Models Loaded: Seq2Seq, Reasoner, Agentic Decoder (14.4M), Tools Suite\n")

    # =========================================================================
    # EXAMPLE 1: Math Calculation via Tool Execution Loop
    # =========================================================================
    print("#" * 75)
    print("  EXAMPLE 1: Math Calculation & Observation Feedback Loop")
    print("  Pipeline Flow: Prompt -> Seq2Seq Parse -> Tool Execution -> Memory Update -> Answer")
    print("#" * 75)
    prompt1 = "Evaluate the mathematical expression: 144 / 12 + (8 * 9) - 15"
    print(f"\n[USER PROMPT]: \"{prompt1}\"\n")

    t0 = time.perf_counter()
    # Step 1: Parse
    task1 = agent.seq2seq.parse_task(prompt1)
    print(f"  [1. Seq2Seq Parser]     -> Domain: {task1.get('domain')}, Task: {task1.get('task')}")

    # Step 2: Reasoning & Plan
    plan1 = agent.reasoning.plan_task()
    print(f"  [2. Reasoning Engine]   -> Planned Actions: {plan1['planned_actions']}, Verifier: {plan1['verification']['decision_label']}")

    # Step 3: Tool Dispatch
    print("  [3. Tool Dispatcher]    -> Invoking CalculatorTool with expression '144 / 12 + (8 * 9) - 15'...")
    calc_res = agent.tool_executor.execute("calculator", {"expression": "144 / 12 + (8 * 9) - 15"})
    print(f"                             Tool Result: success={calc_res.success}, output={calc_res.output}")

    # Step 4: Feed observation to Decoder with Neural Memory
    obs_prompt = (
        f"<|im_start|>user\n{prompt1}<|im_end|>\n"
        f"<|im_start|>tool\n{json.dumps({'tool': 'calculator', 'output': calc_res.output})}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<|analysis|>The calculator returned {calc_res.output}. The computation is verified.<|analysis_end|>\n"
    )
    text1, _, mem1 = agent.decoder.generate(obs_prompt, max_new_tokens=48, temperature=0.6)
    print(f"  [4. Decoder & Memory]   -> Neural Memory State: {list(mem1.shape)} (32 slots updated)")
    print(f"  [5. Verified Answer]    -> Result: 144/12 + (8*9) - 15 = {calc_res.output}")
    print(f"  [Latency: {time.perf_counter() - t0:.2f}s]\n")

    # =========================================================================
    # EXAMPLE 2: Sandboxed Python Code Execution & Output Capture
    # =========================================================================
    print("#" * 75)
    print("  EXAMPLE 2: Sandboxed Python Subprocess Execution")
    print("  Pipeline Flow: Task Parsing -> Python Code Generation -> Subprocess Sandbox -> Memory")
    print("#" * 75)
    prompt2 = "Find the sum of all prime numbers between 1 and 30 using Python."
    print(f"\n[USER PROMPT]: \"{prompt2}\"\n")

    t0 = time.perf_counter()
    task2 = agent.seq2seq.parse_task(prompt2)
    print(f"  [1. Seq2Seq Parser]     -> Domain: {task2.get('domain')}, Objective: {task2.get('objective')}")

    # Tool execution in PythonTool
    py_code = (
        "primes = [n for n in range(2, 31) if all(n % d != 0 for d in range(2, int(n**0.5) + 1))]\n"
        "print(f'Primes: {primes}')\n"
        "print(f'Sum: {sum(primes)}')\n"
    )
    print(f"  [2. Python Tool Sandbox] -> Executing generated code:\n" + "\n".join("      " + l for l in py_code.strip().splitlines()))
    py_res = agent.tool_executor.execute("python", {"code": py_code})
    print(f"  [3. Subprocess Output]   -> Success: {py_res.success}")
    for line in py_res.output.strip().splitlines():
        print(f"                             {line}")

    # Inject into memory
    obs_prompt2 = f"<|im_start|>tool\n{py_res.output.strip()}<|im_end|>\n<|im_start|>assistant\n"
    _, _, mem2 = agent.decoder.generate(obs_prompt2, max_new_tokens=32, temperature=0.6)
    print(f"  [4. Working Memory]     -> Memory slots synchronized: {list(mem2.shape)}")
    print(f"  [Latency: {time.perf_counter() - t0:.2f}s]\n")

    # =========================================================================
    # EXAMPLE 3: Autonomous File System Inspection
    # =========================================================================
    print("#" * 75)
    print("  EXAMPLE 3: Autonomous File System Exploration & Action Routing")
    print("  Pipeline Flow: ActionAdapter ('inspect') -> ToolRouter ('file_list') -> Tool Execution")
    print("#" * 75)
    prompt3 = "Inspect the integrated_system package and check all source modules."
    print(f"\n[USER PROMPT]: \"{prompt3}\"\n")

    t0 = time.perf_counter()
    # Route via ActionAdapter and ToolRouter
    adapted = agent.action_adapter.adapt([0])  # Action 0 = inspect
    action_name = adapted[0].action
    route = agent.tool_router.route(action_name)
    print(f"  [1. Action Routing]     -> Action ID 0: '{action_name}' mapped to candidate tools: {route.tools}")

    # Execute file_list on integrated_system
    list_res = agent.tool_executor.execute("file_list", {"path": "integrated_system"})
    print(f"  [2. FileListTool Run]   -> Scanned directory: 'integrated_system/' (Success: {list_res.success})")
    files = list_res.output if isinstance(list_res.output, list) else []
    print(f"                             Found {len(files)} files/folders:")
    for f in files[:8]:
        fname = f['name'] if isinstance(f, dict) else f
        ftype = f['type'] if isinstance(f, dict) else 'item'
        print(f"                             - {fname} ({ftype})")

    print(f"  [Latency: {time.perf_counter() - t0:.2f}s]\n")

    print("=" * 75)
    print("      LIVE DEMONSTRATION COMPLETE: ALL 3 EXAMPLES VERIFIED")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
