# integrated_system/verify_integration.py
import sys
import io
import time
from pathlib import Path

# Configure utf-8 stdout if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

print("=" * 70)
print("   CHERRY AI: FULL SYSTEM INTEGRATION VERIFICATION")
print("=" * 70)

def test_1_seq2seq():
    print("\n[TEST 1] Testing Seq2Seq Model Loading and Task Parsing...")
    from integrated_system.models.seq2seq import Seq2SeqPipeline
    t0 = time.perf_counter()
    pipeline = Seq2SeqPipeline()
    load_time = time.perf_counter() - t0
    print(f"  [+] Seq2Seq loaded successfully in {load_time:.2f}s (Vocab: {pipeline.vocab_size})")

    prompt = "Can you write a python script to calculate the prime numbers under 100?"
    task = pipeline.parse_task(prompt)
    print(f"  [+] Parsed task domain: {task.get('domain')}")
    print(f"  [+] Parsed objective: {task.get('objective')}")
    return pipeline

def test_2_reasoning():
    print("\n[TEST 2] Testing Reasoning Model, Planner, Search & Verifier...")
    from integrated_system.models.reasoning import ReasoningPipeline
    t0 = time.perf_counter()
    pipeline = ReasoningPipeline()
    load_time = time.perf_counter() - t0
    print(f"  [+] Reasoning model loaded successfully in {load_time:.2f}s")

    result = pipeline.plan_task([2, 15, 34, 88, 4])
    print(f"  [+] Planned actions: {result['planned_actions']}")
    print(f"  [+] Searched actions (Best-First Search): {result['searched_actions']}")
    verif = result["verification"]
    print(f"  [+] Multi-Stage Verifier verdict: {verif['decision_label']} (Safety: {verif['safety']:.2f}, Goal: {verif['goal']:.2f}, State: {verif['state']:.2f})")
    return pipeline

def test_3_tools():
    print("\n[TEST 3] Testing Tools Framework & Execution...")
    tools_dir = WORKSPACE_ROOT / "Self-Perpetuating-Model" / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    from action_adaptor import ActionAdapter
    from tool_router import ToolRouter
    from tool_registry import ToolRegistry
    from tool_executor import ToolExecutor

    adapter = ActionAdapter()
    router = ToolRouter()
    registry = ToolRegistry()
    registry.register_default_tools()
    executor = ToolExecutor(registry)

    # Test ActionAdapter
    actions = adapter.adapt([0, 2, 3, 4, 7])
    action_names = [a.action for a in actions]
    print(f"  [+] ActionAdapter adapted [0, 2, 3, 4, 7] -> {action_names}")

    # Test ToolRouter
    routes = router.route("build")
    print(f"  [+] ToolRouter routed 'build' -> {routes.tools}")

    # Test Calculator tool
    calc_res = executor.execute("calculator", {"expression": "2 ** 10 + 24"})
    assert calc_res.success and calc_res.output == 1048, f"Calc failed: {calc_res}"
    print(f"  [+] Calculator executed: '2 ** 10 + 24' = {calc_res.output}")

    # Test Python tool
    py_res = executor.execute("python", {"code": "print('Cherry Pipeline Online!')"})
    assert py_res.success and "Cherry Pipeline Online!" in py_res.output, f"Python failed: {py_res}"
    print(f"  [+] Python executed successfully: {py_res.output.strip()}")

def test_4_decoder():
    print("\n[TEST 4] Testing Agentic Decoder & Neural Working Memory...")
    from integrated_system.models.decoder import AgenticDecoderPipeline
    t0 = time.perf_counter()
    pipeline = AgenticDecoderPipeline()
    load_time = time.perf_counter() - t0
    print(f"  [+] Decoder model loaded successfully in {load_time:.2f}s (Vocab: {pipeline.model.vocab_size}, d_model: {pipeline.model.d_model})")

    test_prompt = "<|im_start|>user\nCalculate 15 * 12 using the calculator tool.<|im_end|>\n<|im_start|>assistant\n"
    text, tokens, mem_state = pipeline.generate(test_prompt, max_new_tokens=32, temperature=0.7)
    print(f"  [+] Generated tokens: {len(tokens)}")
    print(f"  [+] Neural Working Memory slots state shape: {mem_state.shape}")
    print(f"  [+] Sample generation preview: {repr(text[:100])}")
    return pipeline

def test_5_full_agent():
    print("\n[TEST 5] Testing End-to-End Unified CherryAgent Pipeline...")
    from integrated_system.pipeline import CherryAgent
    t0 = time.perf_counter()
    agent = CherryAgent()
    init_time = time.perf_counter() - t0
    print(f"  [+] Complete CherryAgent initialized in {init_time:.2f}s")

    user_query = "Calculate (45 * 2) + 10 and verify the result."
    print(f"\n  Running prompt: '{user_query}'")
    response = agent.run(user_query, max_tool_turns=3)

    print("\n  --- Pipeline Execution Summary ---")
    print(f"  Domain: {response['task_analysis'].get('domain')}")
    print(f"  Tactical Plan: {response['tactical_plan']}")
    print(f"  Verifier Verdict: {response['verifier_verdict']}")
    print(f"  Tool Calls Made: {len(response['tool_calls'])}")
    for tc in response['tool_calls']:
        print(f"    - Tool: {tc['tool']}, Args: {tc['args']}, Success: {tc['success']}, Output: {tc['output']}")
    print(f"  Final Response Preview:\n    {response['final_response'][:200]}...")

if __name__ == "__main__":
    try:
        test_1_seq2seq()
        test_2_reasoning()
        test_3_tools()
        test_4_decoder()
        test_5_full_agent()
        print("\n" + "=" * 70)
        print("   ALL 5 INTEGRATION VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[!] Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
