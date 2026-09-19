# integrated_system/test_server.py
import sys
import time
from pathlib import Path

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from integrated_system.server import app

client = TestClient(app)

print("=" * 70)
print("      CHERRY AI: FASTAPI SERVER ENDPOINTS TEST")
print("=" * 70)

def test_endpoint_health():
    print("\n[TEST 1] GET /health")
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    print("  [+] Status Code: 200 OK")
    print(f"  [+] Device: {data['device']}")
    print(f"  [+] Available Tools: {data['available_tools']}")
    assert data["status"] == "healthy"

def test_endpoint_parse_task():
    print("\n[TEST 2] POST /parse_task (Seq2Seq Transformer)")
    payload = {"prompt": "Build a machine learning pipeline to predict customer churn"}
    response = client.post("/parse_task", json=payload)
    assert response.status_code == 200
    data = response.json()
    print("  [+] Status Code: 200 OK")
    print(f"  [+] Domain: {data['canonical_task'].get('domain')}")
    print(f"  [+] Objective: {data['canonical_task'].get('objective')}")

def test_endpoint_plan_task():
    print("\n[TEST 3] POST /plan_task (Reasoning Model & Best-First Search)")
    payload = {"task_token_ids": [2, 12, 35, 4]}
    response = client.post("/plan_task", json=payload)
    assert response.status_code == 200
    data = response.json()
    print("  [+] Status Code: 200 OK")
    print(f"  [+] Planned Actions: {data['planned_action_ids']}")
    print(f"  [+] Semantic Actions: {data['semantic_actions']}")
    print(f"  [+] Verifier Verdict: {data['verification']['decision_label']}")

def test_endpoint_execute_tool():
    print("\n[TEST 4] POST /execute_tool (Direct Sandboxed Tool Execution)")
    payload = {"tool": "calculator", "args": {"expression": "25 * 4 + 50"}}
    response = client.post("/execute_tool", json=payload)
    assert response.status_code == 200
    data = response.json()
    print("  [+] Status Code: 200 OK")
    print(f"  [+] Tool Output: {data['output']} (success: {data['success']})")
    assert data["output"] == 150

def test_endpoint_chat():
    print("\n[TEST 5] POST /chat (Complete Autonomous Pipeline)")
    payload = {
        "prompt": "Evaluate (100 / 4) + 15 and confirm the result",
        "max_tool_turns": 2,
        "temperature": 0.5
    }
    t0 = time.perf_counter()
    response = client.post("/chat", json=payload)
    latency = time.perf_counter() - t0
    assert response.status_code == 200
    data = response.json()
    print("  [+] Status Code: 200 OK")
    print(f"  [+] Latency: {latency:.2f}s")
    print(f"  [+] Verifier Verdict: {data['verifier_verdict']}")
    print(f"  [+] Final Response Preview: {data['final_response'][:100]}...")

if __name__ == "__main__":
    try:
        test_endpoint_health()
        test_endpoint_parse_task()
        test_endpoint_plan_task()
        test_endpoint_execute_tool()
        test_endpoint_chat()
        print("\n" + "=" * 70)
        print("      ALL FASTAPI SERVER ENDPOINTS TESTED SUCCESSFULLY!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[!] Server test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
