"""
PostgreSQL adapter for onboarding_answers table.
Handles persistence of LLM-extracted conversational slots.

Migrated from Supabase REST API to direct PostgreSQL (psycopg2) on April 14, 2026.
Uses RECIPROCITY_BACKEND_DB_URL — same connection as supabase_profiles.py.

BUG-022 FIX: Bulletproof persistence with retry + validation
- Retries transient failures (network, connection errors)
- Raises exceptions instead of silent failures
- Validates data was actually persisted
"""
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)


class SupabasePersistenceError(Exception):
    """Raised when slot persistence fails after all retries."""
    pass


class SupabaseOnboardingAdapter:
    """Adapter for persisting onboarding slots to PostgreSQL (formerly Supabase REST)."""

    def __init__(self):
        self.database_url = os.getenv("RECIPROCITY_BACKEND_DB_URL")

        if not self.database_url:
            logger.warning("RECIPROCITY_BACKEND_DB_URL not set - slot persistence disabled")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Onboarding adapter initialized (PostgreSQL direct)")

    def _get_connection(self):
        """Get a new PostgreSQL connection. Caller must close it."""
        return psycopg2.connect(self.database_url)

    def _save_slots_sync(self, user_id: str, slots: List[Dict[str, Any]]) -> int:
        """
        Save multiple slots via PostgreSQL upsert (synchronous).

        Returns number of saved slots.
        Raises SupabasePersistenceError on failure.
        """
        if not slots:
            return 0

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            for slot in slots:
                value = slot.get("value")
                # Convert lists/dicts to string for storage
                if isinstance(value, (list, dict)):
                    import json
                    value = json.dumps(value)
                elif value is not None:
                    value = str(value)

                cursor.execute("""
                    INSERT INTO onboarding_answers (user_id, slot_name, value, confidence, source_text, extraction_method, status, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id, slot_name) DO UPDATE SET
                        value = EXCLUDED.value,
                        confidence = EXCLUDED.confidence,
                        source_text = EXCLUDED.source_text,
                        extraction_method = EXCLUDED.extraction_method,
                        status = EXCLUDED.status,
                        updated_at = NOW()
                """, (
                    user_id,
                    slot.get("name"),
                    value,
                    slot.get("confidence", 1.0),
                    slot.get("source_text"),
                    slot.get("extraction_method", "llm"),
                    slot.get("status", "filled"),
                ))

            conn.commit()
            logger.info(f"✅ Saved {len(slots)} slots for user {user_id[:8]}... to PostgreSQL")
            return len(slots)

        except Exception as e:
            if conn:
                conn.rollback()
            error_msg = f"PostgreSQL save failed for user {user_id[:8]}...: {e}"
            logger.error(error_msg)
            raise SupabasePersistenceError(error_msg)
        finally:
            if conn:
                conn.close()

    def _get_user_slots_sync_internal(self, user_id: str) -> Dict[str, Any]:
        """Fetch all slots for a user (synchronous, internal)."""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT slot_name, value, confidence, status, created_at FROM onboarding_answers WHERE user_id = %s",
                (user_id,)
            )
            rows = cursor.fetchall()
            return {
                row["slot_name"]: {
                    "value": row["value"],
                    "confidence": row["confidence"],
                    "status": row["status"],
                    "created_at": str(row["created_at"]) if row["created_at"] else None,
                }
                for row in rows
            }
        except Exception as e:
            logger.error(f"Error fetching slots for user {user_id[:8]}...: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    # --- Public API (same signatures as before) ---

    async def save_slot(
        self,
        user_id: str,
        slot_name: str,
        value: str,
        confidence: float,
        source_text: Optional[str] = None,
        extraction_method: str = "llm",
        status: str = "filled"
    ) -> bool:
        """Save a single slot (async wrapper)."""
        if not self.enabled:
            return False

        try:
            slot = {
                "name": slot_name,
                "value": value,
                "confidence": confidence,
                "source_text": source_text,
                "extraction_method": extraction_method,
                "status": status,
            }
            result = await asyncio.to_thread(self._save_slots_sync, user_id, [slot])
            return result > 0
        except Exception as e:
            logger.error(f"Error saving slot '{slot_name}': {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((psycopg2.OperationalError, psycopg2.InterfaceError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def save_slots_batch(
        self,
        user_id: str,
        slots: List[Dict[str, Any]]
    ) -> int:
        """
        Save multiple slots in a single batch (upsert) with retry + validation.

        Args:
            user_id: User identifier
            slots: List of slot dicts with keys: name, value, confidence, source_text, etc.

        Returns:
            Number of successfully saved slots

        Raises:
            SupabasePersistenceError: If persistence fails after all retries
        """
        if not self.enabled:
            raise SupabasePersistenceError("Onboarding adapter not enabled - RECIPROCITY_BACKEND_DB_URL not set")

        if not slots:
            return 0

        # Apr-24 fix: drop objective-incompatible slots before save, and run a cleanup sweep
        # of already-stored stale slots once primary_goal is known. Prevents mentorship-specific
        # slots (e.g. "Mentorship Format") surfacing for Service Providers — and the general
        # class of "slot extracted early before primary_goal was set, stays stored even after
        # the filter would exclude it." See filter_slots_by_objective in llm_slot_extractor.py
        # (BUG-045 FIX). This wraps that filter so it applies at the persistence layer too,
        # not just at the LLM extraction layer.
        incoming_count = len(slots)
        slots = await self._drop_objective_incompatible_slots(user_id, slots)
        if not slots:
            logger.info(
                f"save_slots_batch: all {incoming_count} slots filtered out by objective — nothing to save"
            )
            return 0

        count = await asyncio.to_thread(self._save_slots_sync, user_id, slots)

        # Validate persistence
        await self._validate_persistence(user_id, slots)

        # Cleanup sweep: if this batch set/changed primary_goal, delete any previously-stored
        # slots that no longer fit the new objective. Idempotent — safe to run on every save.
        await self._cleanup_stale_slots_by_objective(user_id)

        return count

    async def _drop_objective_incompatible_slots(
        self,
        user_id: str,
        slots: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter out slots that don't fit the user's primary_goal (if known).

        Looks up primary_goal from either this batch (if it's being set now) or
        the existing stored slots. Then runs filter_slots_by_objective to get the
        allowed-slot set and drops any incoming slot not in it. If primary_goal
        is unknown, passes all slots through unchanged (early-conversation grace).
        """
        try:
            # Local import to avoid circular dep at module load time.
            from app.services.llm_slot_extractor import (
                filter_slots_by_objective,
                SLOT_DEFINITIONS,
            )
        except Exception as e:
            logger.warning(f"objective filter unavailable, skipping: {e}")
            return slots

        # Determine primary_goal: batch value wins over stored value (user is updating it).
        primary_goal: Optional[str] = None
        for slot in slots:
            if slot.get("name") == "primary_goal":
                val = slot.get("value")
                if isinstance(val, str) and val.strip():
                    primary_goal = val.strip()
                break
        if not primary_goal:
            try:
                stored = await asyncio.to_thread(self._get_user_slots_sync_internal, user_id)
                pg_val = stored.get("primary_goal", {}).get("value")
                if isinstance(pg_val, str) and pg_val.strip():
                    primary_goal = pg_val.strip()
            except Exception as e:
                logger.debug(f"could not fetch stored primary_goal for filter: {e}")

        if not primary_goal:
            return slots  # Early-conversation grace — no filtering until goal known.

        allowed = set(filter_slots_by_objective(SLOT_DEFINITIONS, primary_goal).keys())
        kept: List[Dict[str, Any]] = []
        dropped: List[str] = []
        for slot in slots:
            name = slot.get("name")
            # primary_goal itself always passes (it's the filter key).
            if name == "primary_goal" or name in allowed:
                kept.append(slot)
            else:
                dropped.append(str(name))
        if dropped:
            logger.info(
                f"[obj-filter] user {user_id[:8]}... goal='{primary_goal}' dropped {len(dropped)} "
                f"incompatible slot(s) from save: {dropped}"
            )
        return kept

    async def _cleanup_stale_slots_by_objective(self, user_id: str) -> None:
        """
        Delete already-stored slots that don't fit the user's current primary_goal.

        Handles the case where a slot (e.g. mentorship_format) was extracted early
        before primary_goal was known, got stored, then primary_goal was later
        resolved to something incompatible (e.g. 'Offer Services'). Without this
        cleanup, the stale slot would remain in the DB and surface in downstream
        consumers (AI summary, focus_slot embeddings, review UI).

        Idempotent: if nothing is stale, runs a cheap SELECT and returns.
        """
        try:
            from app.services.llm_slot_extractor import (
                filter_slots_by_objective,
                SLOT_DEFINITIONS,
            )
        except Exception as e:
            logger.warning(f"objective cleanup unavailable, skipping: {e}")
            return

        try:
            stored = await asyncio.to_thread(self._get_user_slots_sync_internal, user_id)
        except Exception as e:
            logger.debug(f"cleanup could not read stored slots: {e}")
            return

        pg_val = stored.get("primary_goal", {}).get("value") if isinstance(stored.get("primary_goal"), dict) else None
        if not (isinstance(pg_val, str) and pg_val.strip()):
            return  # No goal set yet — nothing to filter against.
        primary_goal = pg_val.strip()

        allowed = set(filter_slots_by_objective(SLOT_DEFINITIONS, primary_goal).keys())
        stale: List[str] = [
            name for name in stored.keys()
            if name != "primary_goal" and name not in allowed
        ]
        if not stale:
            return

        def _delete_stale() -> int:
            conn = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM onboarding_answers WHERE user_id = %s AND slot_name = ANY(%s)",
                    (user_id, stale),
                )
                deleted = cursor.rowcount
                conn.commit()
                return deleted
            except Exception as e:
                if conn:
                    conn.rollback()
                logger.error(f"cleanup delete failed for user {user_id[:8]}...: {e}")
                return 0
            finally:
                if conn:
                    conn.close()

        deleted = await asyncio.to_thread(_delete_stale)
        if deleted:
            logger.info(
                f"[obj-cleanup] user {user_id[:8]}... goal='{primary_goal}' deleted {deleted} "
                f"stale slot(s): {stale}"
            )

    async def _validate_persistence(self, user_id: str, saved_slots: List[Dict[str, Any]]) -> None:
        """Validate that slots were actually persisted to database."""
        try:
            persisted = await self.get_user_slots(user_id)
            saved_slot_names = {slot.get("name") for slot in saved_slots}
            persisted_slot_names = set(persisted.keys())

            missing = saved_slot_names - persisted_slot_names

            if missing:
                error_msg = (
                    f"Persistence validation FAILED for user {user_id[:8]}... "
                    f"Saved {len(saved_slots)} slots but {len(missing)} missing from database: {missing}"
                )
                logger.error(error_msg)
                raise SupabasePersistenceError(error_msg)

            logger.info(f"✅ Persistence validated: {len(saved_slots)} slots confirmed in database")

        except SupabasePersistenceError:
            raise
        except Exception as e:
            error_msg = f"Persistence validation query failed: {e}"
            logger.error(error_msg)
            raise SupabasePersistenceError(error_msg)

    async def get_user_slots(self, user_id: str) -> Dict[str, Any]:
        """Get all slots for a user (async version)."""
        if not self.enabled:
            return {}
        return await asyncio.to_thread(self._get_user_slots_sync_internal, user_id)

    def get_user_slots_sync(self, user_id: str) -> Dict[str, Any]:
        """Get all slots for a user (synchronous version for Celery workers)."""
        if not self.enabled:
            return {}
        return self._get_user_slots_sync_internal(user_id)

    async def get_user_slots_count(self, user_id: str) -> int:
        """Get count of filled slots for a user."""
        if not self.enabled:
            return 0

        try:
            slots = await self.get_user_slots(user_id)
            return len([s for s in slots.values() if s.get("status") in ["filled", "confirmed"]])
        except Exception as e:
            logger.error(f"Error getting slots count: {e}")
            return 0


# Global instance
supabase_onboarding_adapter = SupabaseOnboardingAdapter()
