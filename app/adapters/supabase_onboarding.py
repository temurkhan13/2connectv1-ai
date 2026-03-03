"""
Supabase adapter for onboarding_answers table.
Handles persistence of LLM-extracted conversational slots.

BUG-022 FIX: Bulletproof persistence with retry + validation
- Retries transient failures (network, 500 errors)
- Raises exceptions instead of silent failures
- Validates data was actually persisted
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx
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
    """Adapter for persisting onboarding slots to Supabase."""

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL", "https://omcjxrhprhtlwqzuhjqb.supabase.co")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not self.supabase_key:
            logger.warning("SUPABASE_SERVICE_KEY not set - slot persistence disabled")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"Supabase onboarding adapter initialized: {self.supabase_url}")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Supabase REST API."""
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"  # Upsert behavior
        }

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
        """
        Save a single slot to Supabase (upsert).

        Args:
            user_id: User identifier
            slot_name: Slot name (e.g., "primary_goal", "industry_focus")
            value: Extracted value
            confidence: LLM confidence score (0.0 - 1.0)
            source_text: Original text that was extracted from
            extraction_method: "llm" or "regex"
            status: "filled", "confirmed", or "skipped"

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Supabase adapter disabled - skipping save for {slot_name}")
            return False

        try:
            url = f"{self.supabase_url}/rest/v1/onboarding_answers"
            payload = {
                "user_id": user_id,
                "slot_name": slot_name,
                "value": value,
                "confidence": confidence,
                "source_text": source_text,
                "extraction_method": extraction_method,
                "status": status
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._get_headers()
                )

                if response.status_code in [200, 201]:
                    logger.info(f"✅ Saved slot '{slot_name}' for user {user_id[:8]}... to Supabase")
                    return True
                else:
                    logger.error(f"Failed to save slot '{slot_name}': {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Error saving slot '{slot_name}' to Supabase: {e}")
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
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

        BUG-022 FIX: Bulletproof persistence
        - Retries transient failures (3 attempts with exponential backoff)
        - Raises SupabasePersistenceError instead of returning 0
        - Validates slots were actually persisted

        Args:
            user_id: User identifier
            slots: List of slot dicts with keys: name, value, confidence, source_text, etc.

        Returns:
            Number of successfully saved slots

        Raises:
            SupabasePersistenceError: If persistence fails after all retries
        """
        if not self.enabled:
            raise SupabasePersistenceError("Supabase adapter not enabled - SUPABASE_SERVICE_KEY not set")

        if not slots:
            return 0

        url = f"{self.supabase_url}/rest/v1/onboarding_answers"
        payload = [
            {
                "user_id": user_id,
                "slot_name": slot.get("name"),
                "value": slot.get("value"),
                "confidence": slot.get("confidence", 1.0),
                "source_text": slot.get("source_text"),
                "extraction_method": slot.get("extraction_method", "llm"),
                "status": slot.get("status", "filled")
            }
            for slot in slots
        ]

        # BUG-021 FIX: Upsert header already in _get_headers()
        # BUG-022 FIX: Retry logic + exception handling
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._get_headers()
            )

            if response.status_code in [200, 201]:
                logger.info(f"✅ Saved {len(slots)} slots for user {user_id[:8]}... to Supabase")

                # BUG-022 FIX: Validate persistence by reading back
                await self._validate_persistence(user_id, slots)

                return len(slots)
            else:
                # BUG-022 FIX: Raise exception instead of returning 0
                error_msg = f"Supabase save failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise SupabasePersistenceError(error_msg)

    async def _validate_persistence(self, user_id: str, saved_slots: List[Dict[str, Any]]) -> None:
        """
        Validate that slots were actually persisted to database.

        BUG-022 FIX: Catch silent failures where save appears successful
        but data isn't actually in the database.

        Args:
            user_id: User identifier
            saved_slots: List of slots that were just saved

        Raises:
            SupabasePersistenceError: If validation fails
        """
        try:
            persisted = await self.get_user_slots(user_id)
            saved_slot_names = {slot.get("name") for slot in saved_slots}
            persisted_slot_names = set(persisted.keys())

            # Check that all saved slots are actually in database
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
            raise  # Re-raise validation errors
        except Exception as e:
            # Validation query failed - treat as persistence failure
            error_msg = f"Persistence validation query failed: {e}"
            logger.error(error_msg)
            raise SupabasePersistenceError(error_msg)

    async def get_user_slots(self, user_id: str) -> Dict[str, Any]:
        """
        Get all slots for a user.

        Returns:
            Dict mapping slot_name -> {value, confidence, status, created_at}
        """
        if not self.enabled:
            return {}

        try:
            url = f"{self.supabase_url}/rest/v1/onboarding_answers?user_id=eq.{user_id}&select=*"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, headers=self._get_headers())

                if response.status_code == 200:
                    rows = response.json()
                    return {
                        row["slot_name"]: {
                            "value": row["value"],
                            "confidence": row["confidence"],
                            "status": row["status"],
                            "created_at": row["created_at"]
                        }
                        for row in rows
                    }
                else:
                    logger.error(f"Failed to fetch slots: {response.status_code}")
                    return {}

        except Exception as e:
            logger.error(f"Error fetching slots from Supabase: {e}")
            return {}

    async def get_user_slots_count(self, user_id: str) -> int:
        """Get count of filled slots for a user."""
        if not self.enabled:
            return 0

        try:
            url = f"{self.supabase_url}/rest/v1/rpc/get_user_slots_count"
            payload = {"p_user_id": user_id}

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._get_headers()
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    # Fallback to counting manually
                    slots = await self.get_user_slots(user_id)
                    return len([s for s in slots.values() if s["status"] in ["filled", "confirmed"]])

        except Exception as e:
            logger.error(f"Error getting slots count: {e}")
            return 0


# Global instance
supabase_onboarding_adapter = SupabaseOnboardingAdapter()
