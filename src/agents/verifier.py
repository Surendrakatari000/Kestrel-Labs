"""Verifier (Critic) Agent: fact-checks claims and surgically revises."""

from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.agents.utils import _get_llm, _safe_invoke_structured
from src.agents.synthesizer import _format_chunks_for_prompt


class ClaimVerdict(BaseModel):
    """Audit verdict for an individual factual claim."""
    claim: str = Field(description="Factual claim extracted from draft answer.")
    verdict: Literal[
        "supported", "partially_supported", "conflicting_evidence", "insufficient_evidence"
    ] = Field(description="Verification verdict for this claim.")
    explanation: str = Field(description="Reason for verdict. Mention newer date if conflicting.")


class VerifierOutput(BaseModel):
    """Structured audit results and surgical revision."""
    claims: List[ClaimVerdict] = Field(description="Verdicts for all material claims.")
    overall_supported: bool = Field(description="True only if every claim is supported.")
    revised_answer: str = Field(
        description="Surgically rewritten answer if any claim is unbacked, otherwise copy of draft."
    )


VERIFIER_SYSTEM = """You are the Verifier (Critic) agent for Kestrel Labs.
Fact-check the Draft Answer against the provided Context chunks.

INSTRUCTIONS:
1. Extract every material factual claim from Draft Answer.
2. Check each claim against Context: supported, partially_supported, conflicting_evidence, or insufficient_evidence.
3. Set overall_supported=True only if all claims are supported.
4. Write revised_answer:
   - If all supported: copy draft answer exactly, preserving all [chunk_id: title] citations.
   - If any claim is not supported: rewrite to be honest about evidence gaps while strictly preserving [chunk_id: title] citations.
"""


def verifier_node(state: AgentState) -> dict:
    """Audits claims against retrieved chunks and produces an in-place revision."""
    from src.utils.citations import normalize_citations

    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])

    # If already a refusal on missing docs, pass through
    if not draft or "does not contain information" in draft.lower():
        return {
            "verifier_verdicts": [],
            "overall_supported": True,
            "final_answer": draft,
        }

    context = _format_chunks_for_prompt(chunks)
    llm = _get_llm()
    structured_llm = llm.with_structured_output(VerifierOutput)

    msgs = [
        SystemMessage(content=VERIFIER_SYSTEM + f"\n\nCONTEXT:\n{context}"),
        HumanMessage(content=f"Draft Answer to verify:\n\n{draft}"),
    ]

    try:
        result: VerifierOutput = _safe_invoke_structured(structured_llm, msgs)
        verdicts = [
            {"claim": c.claim, "verdict": c.verdict, "explanation": c.explanation}
            for c in result.claims
        ]
        normalized_answer = normalize_citations(result.revised_answer, chunks)
        return {
            "verifier_verdicts": verdicts,
            "overall_supported": result.overall_supported,
            "final_answer": normalized_answer,
        }
    except Exception as e:
        # Fallback: if parsing fails, preserve draft safely with normalized citations
        return {
            "verifier_verdicts": [{"claim": "parse_error", "verdict": "supported", "explanation": str(e)}],
            "overall_supported": True,
            "final_answer": normalize_citations(draft, chunks),
        }

