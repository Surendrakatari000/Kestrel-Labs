"""
Evaluation runner for the Kestrel Research Assistant.

- Loads 15 questions from results/eval_questions.jsonl
- Runs each through the LangGraph pipeline
- Logs every run to LangSmith (via LANGCHAIN_TRACING_V2=true)
- Writes per-question results to results/eval_results.jsonl
- Writes aggregate metrics to results/metrics_summary.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.graph import build_graph

load_dotenv()

# Ensure LangSmith tracing is on
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "kestrel-research-assistant")

BASE_DIR = Path(__file__).parent
QUESTIONS_PATH = BASE_DIR / "results" / "eval_questions.jsonl"
RESULTS_PATH = BASE_DIR / "results" / "eval_results.jsonl"
METRICS_PATH = BASE_DIR / "results" / "metrics_summary.json"


def extract_citations(text: str) -> list:
    """Extracts chunk_id references from [chunk_id: ...] patterns."""
    pattern = r"\[([a-zA-Z0-9\-]+:[0-9]+)"
    return list(set(re.findall(pattern, text)))


def compute_citation_recall(expected: list, cited: list) -> float:
    """What fraction of expected chunk_ids were actually cited?"""
    if not expected:
        return 1.0 if not cited else 0.5  # No expected = unsupported; partial if it cited something
    expected_set = set(expected)
    cited_set = set(cited)
    overlap = expected_set.intersection(cited_set)
    return len(overlap) / len(expected_set)


def run_single_question(graph, question: dict, history: list = None) -> dict:
    """Runs a single question through the graph and returns results."""
    msgs = list(history) if history else []
    msgs.append(HumanMessage(content=question["question"]))

    state_input = {"messages": msgs, "retry_count": 0}
    start = time.time()

    try:
        final_state = graph.invoke(state_input)
        latency = time.time() - start

        final_answer = final_state.get("final_answer", "")
        verdicts = final_state.get("verifier_verdicts", [])
        chunks = final_state.get("retrieved_chunks", [])
        query_type = final_state.get("query_type", "")
        current_query = final_state.get("current_query", "")

        retrieved_ids = [c["metadata"].get("chunk_id", "") for c in chunks]
        cited_ids = extract_citations(final_answer)

        return {
            "question_id": question["question_id"],
            "question": question["question"],
            "type": question["type"],
            "router_query": current_query,
            "router_type": query_type,
            "retrieved_chunk_ids": retrieved_ids,
            "cited_chunk_ids": cited_ids,
            "expected_chunk_ids": question.get("expected_chunk_ids", []),
            "final_answer": final_answer,
            "verdicts": verdicts,
            "citation_recall": compute_citation_recall(
                question.get("expected_chunk_ids", []), cited_ids
            ),
            "latency_seconds": round(latency, 2),
            "error": None,
        }

    except Exception as e:
        latency = time.time() - start
        return {
            "question_id": question["question_id"],
            "question": question["question"],
            "type": question["type"],
            "final_answer": "",
            "error": str(e),
            "latency_seconds": round(latency, 2),
            "citation_recall": 0.0,
        }


def main():
    print("=" * 60)
    print("Kestrel Research Assistant — Evaluation Suite")
    print("=" * 60)

    graph = build_graph()

    # Load questions
    questions = []
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions.\n")

    results = []
    follow_up_history = {}  # For multi-turn conversations

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['type'].upper():20s} | {q['question'][:60]}...")

        # Handle follow-up chains
        history = []
        depends_on = q.get("depends_on")
        if depends_on and depends_on in follow_up_history:
            history = follow_up_history[depends_on]

        result = run_single_question(graph, q, history)
        results.append(result)

        # Store history for follow-ups
        if q["type"] == "follow_up":
            new_history = list(history) + [
                HumanMessage(content=q["question"]),
                AIMessage(content=result.get("final_answer", "")),
            ]
            follow_up_history[q["question_id"]] = new_history

        status = "✅" if result.get("error") is None else "❌"
        recall = result.get("citation_recall", 0)
        print(f"  {status}  Recall: {recall:.0%}  Latency: {result['latency_seconds']}s")

        # Rate limit safety: small delay between questions
        time.sleep(2)

    # ── Write results ──
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults written to {RESULTS_PATH}")

    # ── Compute aggregate metrics ──
    total = len(results)
    errors = sum(1 for r in results if r.get("error"))
    avg_recall = sum(r["citation_recall"] for r in results) / total if total else 0
    avg_latency = sum(r["latency_seconds"] for r in results) / total if total else 0

    by_type = {}
    for r in results:
        t = r.get("type", "unknown")
        if t not in by_type:
            by_type[t] = {"count": 0, "total_recall": 0, "total_latency": 0}
        by_type[t]["count"] += 1
        by_type[t]["total_recall"] += r["citation_recall"]
        by_type[t]["total_latency"] += r["latency_seconds"]

    type_metrics = {}
    for t, v in by_type.items():
        type_metrics[t] = {
            "count": v["count"],
            "avg_citation_recall": round(v["total_recall"] / v["count"], 3),
            "avg_latency_seconds": round(v["total_latency"] / v["count"], 2),
        }

    metrics = {
        "total_questions": total,
        "errors": errors,
        "avg_citation_recall": round(avg_recall, 3),
        "avg_latency_seconds": round(avg_latency, 2),
        "by_type": type_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {METRICS_PATH}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total questions:     {total}")
    print(f"Errors:              {errors}")
    print(f"Avg citation recall: {avg_recall:.1%}")
    print(f"Avg latency:         {avg_latency:.1f}s")
    for t, m in type_metrics.items():
        print(f"  {t:20s}  recall={m['avg_citation_recall']:.1%}  latency={m['avg_latency_seconds']:.1f}s  (n={m['count']})")
    print("=" * 60)
    print("Check LangSmith for full traces: https://smith.langchain.com")


if __name__ == "__main__":
    main()
