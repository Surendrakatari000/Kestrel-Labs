"""Quick smoke test for the full pipeline."""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: No GROQ_API_KEY found in .env")
    print("Create a .env file with: GROQ_API_KEY=gsk_...")
    exit(1)

print("GROQ_API_KEY found. Testing full pipeline...")

from langchain_core.messages import HumanMessage
from src.graph import build_graph

g = build_graph()
result = g.invoke({"messages": [HumanMessage(content="What is a Beacon?")], "retry_count": 0})

print(f"Query type: {result.get('query_type', '?')}")
print(f"Chunks retrieved: {len(result.get('retrieved_chunks', []))}")
fa = result.get("final_answer", "")
print(f"Final answer (first 300 chars):\n{fa[:300]}")
print("\nSMOKE TEST PASSED!")
