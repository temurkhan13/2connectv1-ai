"""Match-quality counters.

Phase 4 A6 of the [[Apr-28]] admin dashboard audit.

Lightweight Redis-backed counters surfacing matching-pipeline behavior
to the admin dashboard. Counters MUST never break matching — every
write/read is wrapped in try/except.

Key pattern:
    matching:counters:{YYYY-MM-DD}:{counter_name} -> integer (INCR)
    TTL: 90 days (per-key)

Counters tracked (initial set, Apr-28):
    scoring_calls_total       — every _score_pair() invocation
    scoring_parse_failure     — JSONDecodeError caught + fallback path taken
    scoring_score_below_min   — llm_score < LLM_SCORE_MIN (proxy for hard rules
                                like Rule 6 capping reciprocal pairs)
    function_filter_activated — Function Filter narrowed candidate pool
    reciprocity_hard_fired    — Reciprocity Hybrid HARD-mode branch executed
    reciprocity_soft_fired    — Reciprocity Hybrid SOFT-fallback executed
    calibration_floor_applied — Score Calibration Floor adjusted llm_score up

Surfaced via: GET /admin/match-quality-counters?days=30 → returns per-day counts.

See vault Analyses/2026-04-28_admin-dashboard-phase4-design.md for the
deferred-A6 design that motivated this. See Apr-28 session log F/u 11
for ship details.
"""
import os
import logging
from datetime import date, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_redis_client = None
_redis_init_attempted = False

COUNTER_TTL_SECONDS = 90 * 86400  # 90 days per-key; old buckets auto-expire
COUNTER_KEY_PREFIX = "matching:counters"

# Closed allowlist — incr() rejects unknown names so typos can't silently
# pollute the keyspace.
COUNTERS = (
    "scoring_calls_total",
    "scoring_parse_failure",
    "scoring_score_below_min",
    "function_filter_activated",
    "reciprocity_hard_fired",
    "reciprocity_soft_fired",
    "calibration_floor_applied",
)


def _get_redis():
    """Lazy Redis client init. Returns None if Redis unreachable / not configured."""
    global _redis_client, _redis_init_attempted
    if _redis_init_attempted:
        return _redis_client
    _redis_init_attempted = True
    try:
        import redis as redis_mod
        url = os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL")
        if not url:
            logger.info("[match_counters] REDIS_URL not set; counters disabled")
            return None
        _redis_client = redis_mod.from_url(
            url, socket_connect_timeout=2, socket_timeout=2
        )
        _redis_client.ping()
        logger.info("[match_counters] Redis connected; counters active")
        return _redis_client
    except Exception as e:
        logger.warning(f"[match_counters] Redis init failed; counters disabled: {e}")
        _redis_client = None
        return None


def _key(name: str, day: Optional[date] = None) -> str:
    d = (day or date.today()).isoformat()
    return f"{COUNTER_KEY_PREFIX}:{d}:{name}"


def incr(name: str, amount: int = 1) -> None:
    """Increment counter for today. SILENT failure — never raises.

    Wrapped in try/except so a Redis hiccup never breaks the matching path.
    Unknown counter names are rejected (logged, not raised).
    """
    if name not in COUNTERS:
        logger.warning(f"[match_counters] unknown counter '{name}' (rejected)")
        return
    try:
        client = _get_redis()
        if client is None:
            return
        key = _key(name)
        pipe = client.pipeline()
        pipe.incrby(key, amount)
        pipe.expire(key, COUNTER_TTL_SECONDS)
        pipe.execute()
    except Exception as e:
        # Debug-level — don't spam logs if Redis is briefly unavailable
        logger.debug(f"[match_counters] incr({name}) failed: {e}")


def read_counters(days: int = 30) -> Dict[str, Dict[str, int]]:
    """Read all counters for the last N days.

    Returns: {counter_name: {YYYY-MM-DD: count, ...}, ...}
    Silent failure on Redis errors — returns shape with empty per-counter dicts.
    """
    out: Dict[str, Dict[str, int]] = {name: {} for name in COUNTERS}
    try:
        client = _get_redis()
        if client is None:
            return out
        today = date.today()
        # Cap days to TTL — anything older won't exist
        days_capped = max(1, min(days, 90))
        for i in range(days_capped):
            day = today - timedelta(days=i)
            for name in COUNTERS:
                try:
                    val = client.get(_key(name, day))
                    if val is not None:
                        out[name][day.isoformat()] = int(val)
                except Exception:
                    # Skip individual-key errors silently
                    continue
        return out
    except Exception as e:
        logger.warning(f"[match_counters] read_counters failed: {e}")
        return out


def summary(days: int = 30) -> Dict[str, int]:
    """Return total per-counter sum over last N days.

    Convenience for the dashboard summary cards.
    """
    raw = read_counters(days)
    return {name: sum(per_day.values()) for name, per_day in raw.items()}
