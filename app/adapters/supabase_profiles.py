"""
Supabase PostgreSQL adapter for user profiles.

This adapter replaces the DynamoDB adapter (dynamodb.py) for user profile storage.
It provides the same interface as the DynamoDB models to minimize code changes.

Tables:
- user_profiles: Replaces DynamoDB UserProfile
- user_match_cache: Replaces DynamoDB UserMatches
- notified_match_pairs: Replaces DynamoDB NotifiedMatchPairs

Note: Feedback and ChatRecord are handled by existing backend tables
(match_feedback and ai_conversations).
"""

import logging
import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _numpy_safe_default(o):
    """JSON encoder fallback for numpy types.

    Apr-23 fix ([[Apr-22]] F/u 12 audit → Apr-23 F/u 3 investigation):
    the daily `scheduled_matchmaking_task` cron was failing with
    `Object of type float32 is not JSON serializable` since Apr 15 —
    7,197 failed match-cache writes across 142 users. Root cause:
    pgvector's `<=>` cosine op returns similarity scores that flow
    through `find_similar_users` → `format_match_results` → `MatchResult.dict()`
    retaining numpy.float32 identity; when `psycopg2.extras.Json(...)`
    calls `json.dumps()` → TypeError.

    Scoped at the SINK so every `store_user_matches` caller benefits
    (the cron, inline_matching, criteria_matching, future callers).
    This is a [[CODING-DISCIPLINE]] Rule 2 minimum-change fix: one
    defensive coercion at the JSON serialization boundary. No Rule 5
    concern — numpy scalar-type set is closed enum.
    """
    try:
        import numpy as _np
        if isinstance(o, (_np.integer, _np.floating)):
            return o.item()
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _safe_json_dumps(data):
    """json.dumps with numpy-safe fallback for psycopg2 Json wrapper."""
    return json.dumps(data, default=_numpy_safe_default)


class SupabaseProfileAdapter:
    """Base adapter for Supabase profile operations."""

    def __init__(self):
        """Initialize Supabase connection."""
        self.database_url = os.getenv('RECIPROCITY_BACKEND_DB_URL')
        if not self.database_url:
            raise ValueError("RECIPROCITY_BACKEND_DB_URL environment variable is required")

    def get_connection(self):
        """Get database connection. Caller is responsible for closing."""
        return psycopg2.connect(self.database_url)


# Global adapter instance
_adapter: Optional[SupabaseProfileAdapter] = None


def get_adapter() -> SupabaseProfileAdapter:
    """Get or create the global adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = SupabaseProfileAdapter()
    return _adapter


class SupabaseUserProfile:
    """
    Supabase model for user profiles.

    Mirrors the DynamoDB UserProfile interface for compatibility.
    """

    class DoesNotExist(Exception):
        """Raised when a profile is not found."""
        pass

    def __init__(
        self,
        user_id: str,
        resume_link: Optional[str] = None,
        raw_questions: Optional[List[Dict[str, Any]]] = None,
        conversation_text: Optional[str] = None,
        resume_text: Optional[str] = None,
        resume_extracted_at: Optional[datetime] = None,
        resume_extraction_method: Optional[str] = None,
        persona_name: Optional[str] = None,
        persona_archetype: Optional[str] = None,
        persona_designation: Optional[str] = None,
        persona_experience: Optional[str] = None,
        persona_focus: Optional[str] = None,
        persona_profile_essence: Optional[str] = None,
        persona_strategy: Optional[str] = None,
        persona_what_looking_for: Optional[str] = None,
        persona_engagement_style: Optional[str] = None,
        persona_requirements: Optional[str] = None,
        persona_offerings: Optional[str] = None,
        persona_user_type: Optional[str] = None,
        persona_industry: Optional[str] = None,
        persona_generated_at: Optional[datetime] = None,
        processing_status: str = 'not_initiated',
        persona_status: str = 'not_initiated',
        needs_matchmaking: bool = True,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.user_id = user_id
        self.resume_link = resume_link
        self.raw_questions = raw_questions or []
        self.conversation_text = conversation_text or ""
        self.resume_text = resume_text
        self.resume_extracted_at = resume_extracted_at
        self.resume_extraction_method = resume_extraction_method

        # Persona data (nested object for compatibility)
        self._persona_name = persona_name
        self._persona_archetype = persona_archetype
        self._persona_designation = persona_designation
        self._persona_experience = persona_experience
        self._persona_focus = persona_focus
        self._persona_profile_essence = persona_profile_essence
        self._persona_strategy = persona_strategy
        self._persona_what_looking_for = persona_what_looking_for
        self._persona_engagement_style = persona_engagement_style
        self._persona_requirements = persona_requirements
        self._persona_offerings = persona_offerings
        self._persona_user_type = persona_user_type
        self._persona_industry = persona_industry
        self._persona_generated_at = persona_generated_at

        self.processing_status = processing_status
        self.persona_status = persona_status
        self._needs_matchmaking = needs_matchmaking
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)

        # Create nested profile and persona objects for compatibility
        self.profile = _ProfileData(
            resume_link=resume_link,
            raw_questions=raw_questions or [],
            created_at=created_at,
            updated_at=updated_at,
        )
        self.persona = _PersonaData(
            name=persona_name,
            archetype=persona_archetype,
            designation=persona_designation,
            experience=persona_experience,
            focus=persona_focus,
            profile_essence=persona_profile_essence,
            strategy=persona_strategy,
            what_theyre_looking_for=persona_what_looking_for,
            engagement_style=persona_engagement_style,
            requirements=persona_requirements,
            offerings=persona_offerings,
            user_type=persona_user_type,
            industry=persona_industry,
            generated_at=persona_generated_at,
        )
        self.resume_text_data = _ResumeTextData(
            text=resume_text,
            extracted_at=resume_extracted_at,
            extraction_method=resume_extraction_method,
        )

    @property
    def needs_matchmaking(self) -> str:
        """Return needs_matchmaking as string for DynamoDB compatibility."""
        return 'true' if self._needs_matchmaking else 'false'

    @needs_matchmaking.setter
    def needs_matchmaking(self, value):
        """Accept both bool and string values."""
        if isinstance(value, bool):
            self._needs_matchmaking = value
        else:
            self._needs_matchmaking = str(value).lower() == 'true'

    @classmethod
    def get(cls, user_id: str) -> 'SupabaseUserProfile':
        """Get user profile by ID."""
        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM user_profiles WHERE user_id = %s
            """, (user_id,))

            row = cursor.fetchone()
            if not row:
                raise cls.DoesNotExist(f"UserProfile with user_id {user_id} does not exist")

            return cls._from_row(row)

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    @classmethod
    def _from_row(cls, row: Dict[str, Any]) -> 'SupabaseUserProfile':
        """Create instance from database row."""
        return cls(
            user_id=str(row['user_id']),
            resume_link=row.get('resume_link'),
            raw_questions=row.get('raw_questions') or [],
            conversation_text=row.get('conversation_text') or "",
            resume_text=row.get('resume_text'),
            resume_extracted_at=row.get('resume_extracted_at'),
            resume_extraction_method=row.get('resume_extraction_method'),
            persona_name=row.get('persona_name'),
            persona_archetype=row.get('persona_archetype'),
            persona_designation=row.get('persona_designation'),
            persona_experience=row.get('persona_experience'),
            persona_focus=row.get('persona_focus'),
            persona_profile_essence=row.get('persona_profile_essence'),
            persona_strategy=row.get('persona_strategy'),
            persona_what_looking_for=row.get('persona_what_looking_for'),
            persona_engagement_style=row.get('persona_engagement_style'),
            persona_requirements=row.get('persona_requirements'),
            persona_offerings=row.get('persona_offerings'),
            persona_user_type=row.get('persona_user_type'),
            persona_industry=row.get('persona_industry'),
            persona_generated_at=row.get('persona_generated_at'),
            processing_status=row.get('processing_status', 'not_initiated'),
            persona_status=row.get('persona_status', 'not_initiated'),
            needs_matchmaking=row.get('needs_matchmaking', True),
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at'),
        )

    @classmethod
    def create_user(cls, user_id: str, resume_link: Optional[str], questions: List[Dict[str, Any]], conversation_text: str = "") -> 'SupabaseUserProfile':
        """Create a new user profile (does not save to DB yet)."""
        now = datetime.now(timezone.utc)
        raw_questions = [
            {'prompt': q.get('prompt', ''), 'answer': str(q.get('answer', ''))}
            for q in questions
        ]

        return cls(
            user_id=user_id,
            resume_link=resume_link,
            raw_questions=raw_questions,
            conversation_text=conversation_text,
            processing_status='pending',
            persona_status='not_initiated',
            needs_matchmaking=True,
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def scan(cls, limit: Optional[int] = None):
        """Scan all profiles (generator)."""
        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = "SELECT * FROM user_profiles ORDER BY created_at DESC"
            if limit:
                query += f" LIMIT {int(limit)}"

            cursor.execute(query)

            for row in cursor:
                yield cls._from_row(row)

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def save(self) -> bool:
        """Save/update profile in database."""
        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor()

            # Sync nested objects back to flat attributes
            self._sync_from_nested_objects()

            cursor.execute("""
                INSERT INTO user_profiles (
                    user_id, resume_link, raw_questions, conversation_text,
                    resume_text, resume_extracted_at, resume_extraction_method,
                    persona_name, persona_archetype, persona_designation,
                    persona_experience, persona_focus, persona_profile_essence,
                    persona_strategy, persona_what_looking_for, persona_engagement_style,
                    persona_requirements, persona_offerings, persona_user_type,
                    persona_industry, persona_generated_at,
                    processing_status, persona_status, needs_matchmaking,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (user_id) DO UPDATE SET
                    resume_link = EXCLUDED.resume_link,
                    raw_questions = EXCLUDED.raw_questions,
                    conversation_text = EXCLUDED.conversation_text,
                    resume_text = EXCLUDED.resume_text,
                    resume_extracted_at = EXCLUDED.resume_extracted_at,
                    resume_extraction_method = EXCLUDED.resume_extraction_method,
                    persona_name = EXCLUDED.persona_name,
                    persona_archetype = EXCLUDED.persona_archetype,
                    persona_designation = EXCLUDED.persona_designation,
                    persona_experience = EXCLUDED.persona_experience,
                    persona_focus = EXCLUDED.persona_focus,
                    persona_profile_essence = EXCLUDED.persona_profile_essence,
                    persona_strategy = EXCLUDED.persona_strategy,
                    persona_what_looking_for = EXCLUDED.persona_what_looking_for,
                    persona_engagement_style = EXCLUDED.persona_engagement_style,
                    persona_requirements = EXCLUDED.persona_requirements,
                    persona_offerings = EXCLUDED.persona_offerings,
                    persona_user_type = EXCLUDED.persona_user_type,
                    persona_industry = EXCLUDED.persona_industry,
                    persona_generated_at = EXCLUDED.persona_generated_at,
                    processing_status = EXCLUDED.processing_status,
                    persona_status = EXCLUDED.persona_status,
                    needs_matchmaking = EXCLUDED.needs_matchmaking,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                self.user_id,
                self.resume_link,
                Json(self.raw_questions),
                self.conversation_text or "",
                self.resume_text,
                self.resume_extracted_at,
                self.resume_extraction_method,
                self._persona_name,
                self._persona_archetype,
                self._persona_designation,
                self._persona_experience,
                self._persona_focus,
                self._persona_profile_essence,
                self._persona_strategy,
                self._persona_what_looking_for,
                self._persona_engagement_style,
                self._persona_requirements,
                self._persona_offerings,
                self._persona_user_type,
                self._persona_industry,
                self._persona_generated_at,
                self.processing_status,
                self.persona_status,
                self._needs_matchmaking,
                self.created_at,
                datetime.now(timezone.utc),
            ))

            conn.commit()
            logger.info(f"Saved profile for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving profile for user {self.user_id}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def delete(self) -> bool:
        """Delete profile from database."""
        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM user_profiles WHERE user_id = %s", (self.user_id,))
            conn.commit()

            logger.info(f"Deleted profile for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting profile for user {self.user_id}: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _get_real_user_name(self) -> Optional[str]:
        """Fetch the real user name (first_name + last_name) from the users table."""
        try:
            adapter = get_adapter()
            conn = adapter.get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT first_name, last_name FROM users WHERE id = %s::uuid",
                (self.user_id,)
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                name = f"{row[0] or ''} {row[1] or ''}".strip()
                return name if name else None
        except Exception:
            pass
        return None

    def _sync_from_nested_objects(self):
        """Sync attributes from nested objects (profile, persona) to flat fields."""
        if self.profile:
            self.resume_link = self.profile.resume_link
            self.raw_questions = [q.as_dict() if hasattr(q, 'as_dict') else q for q in self.profile.raw_questions]

        if self.persona:
            # Use real user name from users table instead of LLM-generated archetype title
            # The LLM generates names like "The Growth-Focused Founder" — useless for identification
            real_name = self._get_real_user_name()
            self._persona_name = real_name if real_name else self.persona.name
            self._persona_archetype = self.persona.archetype
            self._persona_designation = self.persona.designation
            self._persona_experience = self.persona.experience
            self._persona_focus = self.persona.focus
            self._persona_profile_essence = self.persona.profile_essence
            self._persona_strategy = self.persona.strategy
            self._persona_what_looking_for = self.persona.what_theyre_looking_for
            self._persona_engagement_style = self.persona.engagement_style
            self._persona_requirements = self.persona.requirements
            self._persona_offerings = self.persona.offerings
            self._persona_user_type = self.persona.user_type
            self._persona_industry = self.persona.industry
            self._persona_generated_at = self.persona.generated_at

        if self.resume_text_data:
            self.resume_text = self.resume_text_data.text
            self.resume_extracted_at = self.resume_text_data.extracted_at
            self.resume_extraction_method = self.resume_text_data.extraction_method

    def to_dict(self) -> Dict[str, Any]:
        """Convert user profile to dictionary (same format as DynamoDB)."""
        return {
            "user_id": self.user_id,
            "profile": {
                "resume_link": self.resume_link,
                "raw_questions": self.raw_questions,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            },
            "resume_text": {
                "text": self.resume_text,
                "extracted_at": self.resume_extracted_at.isoformat() if self.resume_extracted_at else None,
                "extraction_method": self.resume_extraction_method,
            } if self.resume_text else None,
            "persona": {
                "name": self._persona_name,
                "archetype": self._persona_archetype,
                "designation": self._persona_designation,
                "experience": self._persona_experience,
                "focus": self._persona_focus,
                "profile_essence": self._persona_profile_essence,
                "strategy": self._persona_strategy,
                "investment_philosophy": self._persona_strategy,  # Legacy field
                "what_theyre_looking_for": self._persona_what_looking_for,
                "engagement_style": self._persona_engagement_style,
                "requirements": self._persona_requirements,
                "offerings": self._persona_offerings,
                "user_type": self._persona_user_type,
                "industry": self._persona_industry,
                "generated_at": self._persona_generated_at.isoformat() if self._persona_generated_at else None,
            } if self._persona_name else None,
            "processing_status": self.processing_status,
            "persona_status": self.persona_status,
            "needs_matchmaking": self.needs_matchmaking
        }


class _ProfileData:
    """Nested profile data object for DynamoDB compatibility."""

    def __init__(
        self,
        resume_link: Optional[str] = None,
        raw_questions: Optional[List[Dict[str, Any]]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.resume_link = resume_link
        self._raw_questions = raw_questions or []
        self.created_at = created_at
        self.updated_at = updated_at

    @property
    def raw_questions(self) -> List:
        """Return raw_questions as list of QuestionAnswer-like objects."""
        return [_QuestionAnswer(q.get('prompt', ''), q.get('answer', '')) for q in self._raw_questions]

    @raw_questions.setter
    def raw_questions(self, value):
        if isinstance(value, list):
            self._raw_questions = [
                q.as_dict() if hasattr(q, 'as_dict') else q
                for q in value
            ]


class _QuestionAnswer:
    """Question/answer pair for DynamoDB compatibility."""

    def __init__(self, prompt: str, answer: str):
        self.prompt = prompt
        self.answer = answer

    def as_dict(self) -> Dict[str, str]:
        return {'prompt': self.prompt, 'answer': self.answer}


class _PersonaData:
    """Nested persona data object for DynamoDB compatibility."""

    def __init__(
        self,
        name: Optional[str] = None,
        archetype: Optional[str] = None,
        designation: Optional[str] = None,
        experience: Optional[str] = None,
        focus: Optional[str] = None,
        profile_essence: Optional[str] = None,
        strategy: Optional[str] = None,
        what_theyre_looking_for: Optional[str] = None,
        engagement_style: Optional[str] = None,
        requirements: Optional[str] = None,
        offerings: Optional[str] = None,
        user_type: Optional[str] = None,
        industry: Optional[str] = None,
        generated_at: Optional[datetime] = None,
    ):
        self.name = name
        self.archetype = archetype
        self.designation = designation
        self.experience = experience
        self.focus = focus
        self.profile_essence = profile_essence
        self.strategy = strategy
        self.investment_philosophy = strategy  # Legacy alias
        self.what_theyre_looking_for = what_theyre_looking_for
        self.engagement_style = engagement_style
        self.requirements = requirements
        self.offerings = offerings
        self.user_type = user_type
        self.industry = industry
        self.generated_at = generated_at


class _ResumeTextData:
    """Nested resume text data for DynamoDB compatibility."""

    def __init__(
        self,
        text: Optional[str] = None,
        extracted_at: Optional[datetime] = None,
        extraction_method: Optional[str] = None,
    ):
        self.text = text
        self.extracted_at = extracted_at
        self.extraction_method = extraction_method


class SupabaseUserMatches:
    """
    Supabase model for user match cache.

    Mirrors the DynamoDB UserMatches interface for compatibility.
    """

    class DoesNotExist(Exception):
        """Raised when matches are not found."""
        pass

    def __init__(
        self,
        user_id: str,
        matches: Optional[Dict[str, Any]] = None,
        total_matches: int = 0,
        algorithm: Optional[str] = None,
        threshold: Optional[float] = None,
        last_updated: Optional[datetime] = None,
        created_at: Optional[datetime] = None,
    ):
        self.user_id = user_id
        self.matches = matches or {}
        self.total_matches = total_matches
        self.algorithm = algorithm
        self.threshold = threshold
        self.last_updated = last_updated or datetime.now(timezone.utc)
        self.created_at = created_at or datetime.now(timezone.utc)

    @classmethod
    def get(cls, user_id: str) -> 'SupabaseUserMatches':
        """Get user matches by ID."""
        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM user_match_cache WHERE user_id = %s
            """, (user_id,))

            row = cursor.fetchone()
            if not row:
                raise cls.DoesNotExist(f"UserMatches with user_id {user_id} does not exist")

            return cls(
                user_id=str(row['user_id']),
                matches=row.get('matches') or {},
                total_matches=row.get('total_matches', 0),
                algorithm=row.get('algorithm'),
                threshold=float(row['threshold']) if row.get('threshold') else None,
                last_updated=row.get('last_updated'),
                created_at=row.get('created_at'),
            )

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    @classmethod
    def store_user_matches(cls, user_id: str, matches_data: Dict[str, Any]) -> bool:
        """Store matches for a user."""
        if not user_id:
            logger.warning("store_user_matches called with empty user_id")
            return False

        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor()

            total_matches = len(matches_data.get('requirements_matches', [])) + \
                           len(matches_data.get('offerings_matches', []))

            cursor.execute("""
                INSERT INTO user_match_cache (user_id, matches, total_matches, last_updated)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO UPDATE SET
                    matches = EXCLUDED.matches,
                    total_matches = EXCLUDED.total_matches,
                    last_updated = CURRENT_TIMESTAMP
            """, (user_id, Json(matches_data, dumps=_safe_json_dumps), total_matches))

            conn.commit()
            logger.info(f"Stored {total_matches} matches for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing matches for user {user_id}: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    @classmethod
    def get_user_matches(cls, user_id: str) -> Optional[Dict[str, Any]]:
        """Get stored matches for a user."""
        if not user_id:
            logger.warning("get_user_matches called with empty user_id")
            return None

        try:
            user_matches = cls.get(user_id)
            matches = user_matches.matches

            return {
                'user_id': user_id,
                'total_matches': user_matches.total_matches,
                'requirements_matches': matches.get('requirements_matches', []),
                'offerings_matches': matches.get('offerings_matches', []),
                'last_updated': user_matches.last_updated.isoformat() if user_matches.last_updated else None
            }
        except cls.DoesNotExist:
            return {
                'user_id': user_id,
                'total_matches': 0,
                'requirements_matches': [],
                'offerings_matches': [],
                'last_updated': None
            }
        except Exception as e:
            logger.error(f"Error getting matches for user {user_id}: {e}")
            return None

    @classmethod
    def get_all_user_matches(cls) -> Dict[str, Dict[str, Any]]:
        """
        Bulk fetch all user matches in a single query.

        Returns:
            Dict mapping user_id to their matches data.
            Empty dict on error.
        """
        adapter = get_adapter()
        conn = None
        cursor = None
        result = {}

        try:
            conn = adapter.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT user_id, matches, total_matches, last_updated
                FROM user_match_cache
            """)

            for row in cursor.fetchall():
                uid = str(row['user_id'])
                matches = row.get('matches') or {}
                result[uid] = {
                    'user_id': uid,
                    'total_matches': row.get('total_matches', 0),
                    'requirements_matches': matches.get('requirements_matches', []),
                    'offerings_matches': matches.get('offerings_matches', []),
                    'last_updated': row['last_updated'].isoformat() if row.get('last_updated') else None
                }

            logger.info(f"Bulk fetched {len(result)} user match caches")
            return result

        except Exception as e:
            logger.error(f"Error bulk fetching user matches: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    @classmethod
    def clear_user_matches(cls, user_id: str) -> bool:
        """Clear all stored matches for a user."""
        if not user_id:
            logger.warning("clear_user_matches called with empty user_id")
            return False

        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM user_match_cache WHERE user_id = %s", (user_id,))
            conn.commit()

            logger.info(f"Cleared matches for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error clearing matches for user {user_id}: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


class SupabaseNotifiedMatchPairs:
    """
    Supabase model for notified match pairs.

    Mirrors the DynamoDB NotifiedMatchPairs interface for compatibility.
    """

    class DoesNotExist(Exception):
        """Raised when pair is not found."""
        pass

    @staticmethod
    def _order_users(user_id_1: str, user_id_2: str) -> tuple:
        """Ensure consistent ordering (a < b)."""
        if user_id_1 < user_id_2:
            return user_id_1, user_id_2
        return user_id_2, user_id_1

    @classmethod
    def is_pair_notified(cls, user_id_1: str, user_id_2: str) -> bool:
        """Check if this pair has already been notified."""
        if not user_id_1 or not user_id_2:
            return False

        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor()

            user_a, user_b = cls._order_users(user_id_1, user_id_2)

            cursor.execute("""
                SELECT 1 FROM notified_match_pairs
                WHERE user_a_id = %s AND user_b_id = %s
            """, (user_a, user_b))

            return cursor.fetchone() is not None

        except Exception as e:
            logger.error(f"Error checking notified pair: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    @classmethod
    def mark_pair_notified(cls, user_id_1: str, user_id_2: str, similarity_score: float = None) -> bool:
        """Mark a pair as notified (or increment count if exists)."""
        if not user_id_1 or not user_id_2:
            return False

        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor()

            user_a, user_b = cls._order_users(user_id_1, user_id_2)

            cursor.execute("""
                INSERT INTO notified_match_pairs (user_a_id, user_b_id, last_similarity_score)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_a_id, user_b_id) DO UPDATE SET
                    notification_count = notified_match_pairs.notification_count + 1,
                    notified_at = CURRENT_TIMESTAMP,
                    last_similarity_score = COALESCE(EXCLUDED.last_similarity_score, notified_match_pairs.last_similarity_score)
            """, (user_a, user_b, similarity_score))

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error marking pair notified: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    @classmethod
    def get_user_notified_count(cls, user_id: str) -> int:
        """Get how many unique pairs this user has been notified about."""
        if not user_id:
            return 0

        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM notified_match_pairs
                WHERE user_a_id = %s OR user_b_id = %s
            """, (user_id, user_id))

            result = cursor.fetchone()
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Error getting user notified count: {e}")
            return 0
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    @classmethod
    def clear_user_pairs(cls, user_id: str) -> int:
        """Clear all notified pairs involving this user."""
        if not user_id:
            return 0

        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM notified_match_pairs
                WHERE user_a_id = %s OR user_b_id = %s
            """, (user_id, user_id))

            deleted_count = cursor.rowcount
            conn.commit()

            logger.info(f"Cleared {deleted_count} notified pairs for user {user_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error clearing notified pairs for user {user_id}: {e}")
            return 0
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    @classmethod
    def scan(cls):
        """Scan all pairs (generator)."""
        adapter = get_adapter()
        conn = None
        cursor = None
        try:
            conn = adapter.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("SELECT * FROM notified_match_pairs")

            for row in cursor:
                yield _NotifiedPairRow(
                    user_a_id=str(row['user_a_id']),
                    user_b_id=str(row['user_b_id']),
                    notified_at=row.get('notified_at'),
                    notification_count=row.get('notification_count', 1),
                    last_similarity_score=row.get('last_similarity_score'),
                )

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    # Legacy method for DynamoDB compatibility
    @staticmethod
    def generate_pair_key(user_id_1: str, user_id_2: str) -> Optional[str]:
        """Generate consistent pair key (legacy compatibility)."""
        if not user_id_1 or not user_id_2:
            return None
        sorted_ids = sorted([user_id_1, user_id_2])
        return f"{sorted_ids[0]}_{sorted_ids[1]}"


class _NotifiedPairRow:
    """Row object for notified pairs scan."""

    def __init__(
        self,
        user_a_id: str,
        user_b_id: str,
        notified_at: Optional[datetime] = None,
        notification_count: int = 1,
        last_similarity_score: Optional[float] = None,
    ):
        self.user_a_id = user_a_id
        self.user_b_id = user_b_id
        self.notified_at = notified_at
        self.notification_count = notification_count
        self.last_similarity_score = last_similarity_score
        # Legacy compatibility
        self.pair_key = f"{user_a_id}_{user_b_id}"

    def delete(self) -> bool:
        """Delete this pair."""
        return SupabaseNotifiedMatchPairs.clear_user_pairs(self.user_a_id) > 0


# Aliases for drop-in replacement
UserProfile = SupabaseUserProfile
UserMatches = SupabaseUserMatches
NotifiedMatchPairs = SupabaseNotifiedMatchPairs

# Helper class aliases for compatibility with DynamoDB imports
QuestionAnswer = _QuestionAnswer
PersonaData = _PersonaData
ProfileData = _ProfileData
ResumeTextData = _ResumeTextData
