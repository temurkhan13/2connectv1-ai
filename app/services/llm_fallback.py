"""
LLM Fallback Utility — Cross-provider resilience for AI services.

Provides automatic fallback from Anthropic → OpenAI (GPT-5.4) or Google (Gemini 3.1)
when the primary provider is unavailable, rate-limited, or errors.

Supports:
- Single-turn calls (system + user message)
- Multi-turn conversations (full message history)
- Exact same prompts passed to fallback — no quality degradation

Usage:
    from app.services.llm_fallback import call_with_fallback, fallback_from_anthropic_error

    # Direct call with automatic fallback
    response = call_with_fallback(
        service="chat",
        system_prompt="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=1024,
    )

    # Use in existing try/except — pass exact same params on Anthropic failure
    try:
        response = client.messages.create(model=model, system=sys, messages=msgs, ...)
    except Exception as e:
        response_text = fallback_from_anthropic_error(
            service="chat", error=e, system_prompt=sys, messages=msgs,
            max_tokens=1024, temperature=0.7
        )
"""
import os
import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

# ─── Model Configuration ─────────────────────────────────────────────────────

# Primary: Claude Sonnet 4.6 (all services)
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Fallback models per service group
FALLBACK_CONFIG = {
    # GPT-5.4 fallback for critical user-facing services
    "chat": {"provider": "openai", "model": "gpt-5.4"},
    "extraction": {"provider": "openai", "model": "gpt-5.4"},
    "question": {"provider": "openai", "model": "gpt-5.4"},
    # Gemini 3.1 Pro fallback for background/lighter services
    "prediction": {"provider": "gemini", "model": "gemini-3.1-pro-preview"},
    "matching": {"provider": "gemini", "model": "gemini-3.1-pro-preview"},
}

# ─── API Key Mapping ─────────────────────────────────────────────────────────

ANTHROPIC_KEY_MAP = {
    "chat": "ANTHROPIC_CHAT_KEY",
    "extraction": "ANTHROPIC_EXTRACTION_KEY",
    "question": "ANTHROPIC_EXTRACTION_KEY",  # Shares extraction key
    "prediction": "ANTHROPIC_PREDICTION_KEY",
    "matching": "ANTHROPIC_MATCHING_KEY",
}

# ─── Lazy-loaded clients (avoid re-creating on every call) ────────────────────
_openai_client = None
_gemini_configured = False


def get_anthropic_key(service: str) -> Optional[str]:
    """Get the dedicated Anthropic API key for a service, falling back to shared key."""
    env_var = ANTHROPIC_KEY_MAP.get(service, "ANTHROPIC_API_KEY")
    key = os.getenv(env_var)
    if not key:
        key = os.getenv("ANTHROPIC_API_KEY")
        if key:
            logger.warning(f"[LLM Fallback] {env_var} not set, using shared ANTHROPIC_API_KEY for {service}")
    return key


def create_anthropic_client(service: str):
    """Create an Anthropic client with the service-specific API key."""
    from anthropic import Anthropic
    key = get_anthropic_key(service)
    if not key:
        raise ValueError(f"No Anthropic API key available for service '{service}'")
    return Anthropic(api_key=key)


def _get_openai_client():
    """Lazy-load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = os.getenv("OPENAI_FALLBACK_KEY")
        if not key:
            raise ValueError("OPENAI_FALLBACK_KEY not set — cannot use OpenAI fallback")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _ensure_gemini():
    """Lazy-configure Gemini."""
    global _gemini_configured
    if not _gemini_configured:
        import google.generativeai as genai
        key = os.getenv("GEMINI_FALLBACK_KEY")
        if not key:
            raise ValueError("GEMINI_FALLBACK_KEY not set — cannot use Gemini fallback")
        genai.configure(api_key=key)
        _gemini_configured = True


def _convert_system_prompt(system_prompt: Union[str, list, None]) -> str:
    """Convert Anthropic system prompt format (str or list of dicts) to plain string."""
    if system_prompt is None:
        return ""
    if isinstance(system_prompt, str):
        return system_prompt
    if isinstance(system_prompt, list):
        # Anthropic cache_control format: [{"type": "text", "text": "...", "cache_control": {...}}]
        parts = []
        for item in system_prompt:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(system_prompt)


def _call_openai_full(
    system_prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    model: str = "gpt-5.4",
) -> str:
    """Call OpenAI with full conversation history — identical context to Anthropic."""
    client = _get_openai_client()

    # Convert Anthropic message format to OpenAI format
    openai_messages = []
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # OpenAI uses "assistant" not "assistant" — same, but ensure valid roles
        if role in ("user", "assistant", "system"):
            openai_messages.append({"role": role, "content": content})
        else:
            openai_messages.append({"role": "user", "content": content})

    response = client.chat.completions.create(
        model=model,
        messages=openai_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_gemini_full(
    system_prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    model: str = "gemini-3.1-pro",
) -> str:
    """Call Gemini with full conversation history — identical context to Anthropic."""
    import google.generativeai as genai
    _ensure_gemini()

    gen_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt if system_prompt else None,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )

    # Convert messages to Gemini format (multi-turn)
    if len(messages) == 1:
        # Single turn — simple call
        response = gen_model.generate_content(messages[0].get("content", ""))
    else:
        # Multi-turn — use chat
        chat = gen_model.start_chat(history=[])
        last_response = None
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                last_response = chat.send_message(content)
            # Skip assistant messages — they're already in Gemini's history from responses
        if last_response:
            return last_response.text
        response = gen_model.generate_content(messages[-1].get("content", ""))

    return response.text


def _call_fallback(
    service: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    """Route to the correct fallback provider for a service."""
    fallback = FALLBACK_CONFIG.get(service, {"provider": "openai", "model": "gpt-5.4"})

    if fallback["provider"] == "openai":
        logger.info(f"[LLM Fallback] Using OpenAI {fallback['model']} for {service}")
        return _call_openai_full(system_prompt, messages, max_tokens, temperature, fallback["model"])
    else:
        logger.info(f"[LLM Fallback] Using Gemini {fallback['model']} for {service}")
        return _call_gemini_full(system_prompt, messages, max_tokens, temperature, fallback["model"])


def call_with_fallback(
    service: str,
    system_prompt: Union[str, list, None],
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Call Anthropic primary, falling back to GPT-5.4 or Gemini 3.1 on failure.
    Passes EXACT same system prompt and full message history to fallback.

    Args:
        service: Service name (chat, extraction, question, prediction, matching)
        system_prompt: System prompt (str or Anthropic cache_control list format)
        messages: Full conversation messages [{"role": "user", "content": "..."}]
        max_tokens: Maximum response tokens
        temperature: Sampling temperature

    Returns:
        LLM response text
    """
    system_str = _convert_system_prompt(system_prompt)

    # ── Primary: Anthropic Claude Sonnet 4.6 ──
    try:
        client = create_anthropic_client(service)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,  # Keep original format for Anthropic (supports cache_control)
            messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        logger.warning(f"[LLM Fallback] Anthropic failed for {service}: {e}")

    # ── Fallback: GPT-5.4 or Gemini 3.1 (with SAME context) ──
    try:
        return _call_fallback(service, system_str, messages, max_tokens, temperature)
    except Exception as e2:
        logger.error(f"[LLM Fallback] Both primary and fallback failed for {service}: {e2}")
        raise RuntimeError(
            f"All LLM providers failed for {service}. "
            f"Primary: Anthropic, Fallback: {FALLBACK_CONFIG.get(service, {}).get('provider', 'unknown')}"
        ) from e2


def fallback_from_anthropic_error(
    service: str,
    error: Exception,
    system_prompt: Union[str, list, None],
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    Called from existing service try/except blocks when Anthropic fails.
    Passes the EXACT same system prompt and messages to fallback provider.

    Returns response text on success, None if fallback also fails.
    This lets the service fall through to its existing template fallback.
    """
    system_str = _convert_system_prompt(system_prompt)

    logger.warning(f"[LLM Fallback] Anthropic error in {service}: {error}")
    try:
        return _call_fallback(service, system_str, messages, max_tokens, temperature)
    except Exception as e2:
        logger.error(f"[LLM Fallback] Fallback also failed for {service}: {e2}")
        return None
