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
    """Executes LLM call with bounded exponential backoff and quota fallback."""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _call():
        return llm.invoke(messages)

    try:
        return _call()
    except Exception as e:
        err = str(e).lower()
        if "tokens per day" in err or "tpd" in err:
            fallback = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=2048)
            return fallback.invoke(messages)
        raise


def _safe_invoke_structured(structured_llm, messages):
    """Executes structured LLM call with bounded exponential backoff."""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        retry=retry_if_exception_type(Exception),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _call():
        return structured_llm.invoke(messages)

    try:
        return _call()
    except Exception as e:
        err = str(e).lower()
        if "tokens per day" in err or "tpd" in err:
            # Recreate with 20b model fallback using same output schema
            schema = getattr(structured_llm, "schema", None) or getattr(structured_llm, "_schema", None)
            if schema:
                fallback = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=2048).with_structured_output(schema)
                return fallback.invoke(messages)
        raise
