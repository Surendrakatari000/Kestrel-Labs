"""LLM setup with rate-limit exponential backoff."""

import os
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def _get_llm(temperature: float = 0, model: str = None, max_tokens: int = 2048):
    """Initializes ChatGroq with output token limits."""
    model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
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
        if "tokens per day" in err or "tpd" in err or "does not exist" in err or "404" in err:
            # Fallback model with available daily token quota
            alt_model = "qwen/qwen3.8-27b" if "gpt-oss" in str(getattr(llm, "model_name", "")) else "openai/gpt-oss-20b"
            fallback = ChatGroq(model=alt_model, temperature=0, max_tokens=1024)
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
        if "tokens per day" in err or "tpd" in err or "does not exist" in err or "404" in err:
            schema = getattr(structured_llm, "schema", None) or getattr(structured_llm, "_schema", None)
            if schema:
                alt_model = "qwen/qwen3.8-27b"
                fallback = ChatGroq(model=alt_model, temperature=0, max_tokens=512).with_structured_output(schema)
                return fallback.invoke(messages)
        raise

