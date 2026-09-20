"""LLM setup with rate-limit exponential backoff."""

import os
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def _get_llm(temperature: float = 0):
    """Initializes ChatGroq with output token limits."""
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=2048,
    )


def _log_retry(retry_state):
    """Logs rate-limit backoff events."""
    print(
        f"[RateLimit] HTTP 429: Retrying in {retry_state.next_action.sleep:.1f}s...",
        flush=True,
    )


def _safe_invoke(llm, messages):
    """Executes LLM call with bounded exponential backoff (4s -> 60s)."""
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _call():
        return llm.invoke(messages)
    return _call()


def _safe_invoke_structured(structured_llm, messages):
    """Executes structured LLM call with bounded exponential backoff."""
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _call():
        return structured_llm.invoke(messages)
    return _call()
