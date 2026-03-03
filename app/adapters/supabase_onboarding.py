"""
Supabase adapter for onboarding_answers table.
Handles persistence of LLM-extracted conversational slots.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


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

    async def save_slots_batch(
        self,
        user_id: str,
        slots: List[Dict[str, Any]]
    ) -> int:
        """
        Save multiple slots in a single batch (upsert).

        Args:
            user_id: User identifier
            slots: List of slot dicts with keys: name, value, confidence, source_text, etc.

        Returns:
            Number of successfully saved slots
        """
        if not self.enabled or not slots:
            return 0

        try:
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

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._get_headers()
                )

                if response.status_code in [200, 201]:
                    logger.info(f"✅ Saved {len(slots)} slots for user {user_id[:8]}... to Supabase")
                    return len(slots)
                else:
                    logger.error(f"Failed to save batch: {response.status_code} - {response.text}")
                    return 0

        except Exception as e:
            logger.error(f"Error saving batch to Supabase: {e}")
            return 0

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
