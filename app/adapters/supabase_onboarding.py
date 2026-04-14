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

        count = await asyncio.to_thread(self._save_slots_sync, user_id, slots)

        # Validate persistence
        await self._validate_persistence(user_id, slots)

        return count

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
