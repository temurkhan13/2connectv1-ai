"""
LLM Fallback Utility — Cross-provider resilience for AI services.

Provides automatic fallback from Anthropic → OpenAI (GPT-5.4) or Google (Gemini 3.1)
when the primary provider is unavailable, rate-limited, or errors.

Usage:
    from app.services.llm_fallback import create_llm_client, call_with_fallback

    # Get configured client for a specific service
    client = create_llm_client("chat")  # Uses ANTHROPIC_CHAT_KEY

    # Call with automatic fallback
    response = call_with_fallback(
        service="chat",
        system_prompt="You are a helpful assistant.",
        user_message="Hello!",
        max_tokens=1024,
    )
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ─── Model Configuration ─────────────────────────────────────────────────────

# Primary: Claude Sonnet 4.6 (all services)
ANTHROPIC_MODEL = "claude-sonnet-4-6-20250929"

# Fallback models per service group
FALLBACK_CONFIG = {
    # GPT-5.4 fallback for critical user-facing services
    "chat": {"provider": "openai", "model": "gpt-5.4"},
    "extraction": {"provider": "openai", "model": "gpt-5.4"},
    "question": {"provider": "openai", "model": "gpt-5.4"},
    # Gemini 3.1 fallback for background/lighter services
    "prediction": {"provider": "gemini", "model": "gemini-3.1-pro"},
    "matching": {"provider": "gemini", "model": "gemini-3.1-pro"},
}

# ─── API Key Mapping ─────────────────────────────────────────────────────────

ANTHROPIC_KEY_MAP = {
    "chat": "ANTHROPIC_CHAT_KEY",
    "extraction": "ANTHROPIC_EXTRACTION_KEY",
    "question": "ANTHROPIC_EXTRACTION_KEY",  # Shares extraction key
    "prediction": "ANTHROPIC_PREDICTION_KEY",
    "matching": "ANTHROPIC_MATCHING_KEY",
}


def get_anthropic_key(service: str) -> Optional[str]:
    """Get the dedicated Anthropic API key for a service, falling back to shared key."""
    env_var = ANTHROPIC_KEY_MAP.get(service, "ANTHROPIC_API_KEY")
    key = os.getenv(env_var)
    if not key:
        # Fall back to shared key if dedicated key not set
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


def _call_openai(system_prompt: str, user_message: str, max_tokens: int = 1024,
                 temperature: float = 0.7, model: str = "gpt-5.4") -> str:
    """Call OpenAI GPT as fallback."""
    from openai import OpenAI
    key = os.getenv("OPENAI_FALLBACK_KEY")
    if not key:
        raise ValueError("OPENAI_FALLBACK_KEY not set — cannot use OpenAI fallback")
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt: str, user_message: str, max_tokens: int = 1024,
                 temperature: float = 0.7, model: str = "gemini-3.1-pro") -> str:
    """Call Google Gemini as fallback."""
    import google.generativeai as genai
    key = os.getenv("GEMINI_FALLBACK_KEY")
    if not key:
        raise ValueError("GEMINI_FALLBACK_KEY not set — cannot use Gemini fallback")
    genai.configure(api_key=key)
    gen_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    response = gen_model.generate_content(user_message)
    return response.text


def call_with_fallback(
    service: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Call Anthropic primary, falling back to GPT-5.4 or Gemini 3.1 on failure.

    Args:
        service: Service name (chat, extraction, question, prediction, matching)
        system_prompt: System prompt for the LLM
        user_message: User message / input
        max_tokens: Maximum response tokens
        temperature: Sampling temperature

    Returns:
        LLM response text
    """
    # ── Primary: Anthropic Claude Sonnet 4.6 ──
    try:
        client = create_anthropic_client(service)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
    except Exception as e:
        logger.warning(f"[LLM Fallback] Anthropic failed for {service}: {e}")

    # ── Fallback: GPT-5.4 or Gemini 3.1 ──
    fallback = FALLBACK_CONFIG.get(service, {"provider": "openai", "model": "gpt-5.4"})
    try:
        if fallback["provider"] == "openai":
            logger.info(f"[LLM Fallback] Falling back to OpenAI {fallback['model']} for {service}")
            return _call_openai(system_prompt, user_message, max_tokens, temperature, fallback["model"])
        else:
            logger.info(f"[LLM Fallback] Falling back to Gemini {fallback['model']} for {service}")
            return _call_gemini(system_prompt, user_message, max_tokens, temperature, fallback["model"])
    except Exception as e2:
        logger.error(f"[LLM Fallback] Both primary and fallback failed for {service}: {e2}")
        raise RuntimeError(f"All LLM providers failed for {service}. Primary: Anthropic, Fallback: {fallback['provider']}") from e2
