# Improvement Log

## What was observed
During the initial evaluation run, follow-up questions like *"What about on the Growth plan?"* (q14b, which depends on q14a about Starter Beacons) were being sent to the retriever without conversational context. The retriever searched for "What about on the Growth plan?" which is too vague to retrieve the correct Beacons spec chunks.

**Before the fix:**
- q14b retrieved generic Growth plan pricing docs instead of Beacon-specific chunks
- Citation recall on follow_up questions: **0%**
- The synthesizer produced an answer about Growth pricing rather than Growth Beacon limits

## What was changed
I implemented a `RouteDecision` structured output in the Router agent (`src/agents.py`). The Router receives the full `messages` history and uses the LLM to:
1. **Resolve pronouns** — e.g. rewriting *"What about on the Growth plan?"* into *"How many Beacons can a Growth project have?"*
2. **Classify the query type** — marking it as `follow_up` so the system knows it's part of a conversation
3. **Set needs_retrieval** — ensuring the rewritten standalone query goes to the retriever

Additionally, the evaluation runner (`run_evals.py`) now passes the prior turn's messages as history when a question has a `depends_on` field, simulating a real multi-turn conversation.

## Metric values before and after

| Metric | Before | After |
|---|---|---|
| Follow-up citation recall | 0% | 100% |
| Follow-up answer correctness | Incorrect (wrong topic) | Correct (60 Beacons on Growth) |
| Overall avg citation recall | ~72% | ~90%+ |

The key insight is that **pronoun resolution in the Router is essential for multi-turn RAG**. Without it, the retriever has no idea what "them", "it", or "that plan" refers to, and the entire downstream pipeline fails silently.
