"""
Interactive Persona Approval Service.

Manages the review and approval workflow for generated personas.
Users can review, edit sections, and approve before matching begins.

Key features:
1. Section-by-section persona review
2. Inline editing with AI regeneration option
3. Version tracking for changes
4. Confidence scoring for each section
5. Final approval workflow
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Status of persona approval."""
    DRAFT = "draft"              # Initial generated state
    IN_REVIEW = "in_review"      # User is reviewing
    PENDING_CHANGES = "pending_changes"  # Changes requested
    APPROVED = "approved"        # User approved
    REJECTED = "rejected"        # User rejected completely


class SectionType(str, Enum):
    """Types of persona sections."""
    IDENTITY = "identity"        # Name, archetype, designation
    PROFILE = "profile"          # Profile essence, focus
    PHILOSOPHY = "philosophy"    # Investment philosophy
    PREFERENCES = "preferences"  # What they're looking for
    STYLE = "style"              # Engagement style
    REQUIREMENTS = "requirements"  # Requirements text
    OFFERINGS = "offerings"      # Offerings text


class EditSource(str, Enum):
    """Source of an edit."""
    USER = "user"                # User manually edited
    AI_REGEN = "ai_regen"        # AI regenerated
    SYSTEM = "system"            # System adjustment


@dataclass
class SectionEdit:
    """Record of a section edit."""
    edit_id: str
    section: SectionType
    field_name: str
    old_value: Any
    new_value: Any
    edit_source: EditSource
    timestamp: datetime
    reason: Optional[str] = None


@dataclass
class PersonaSection:
    """A reviewable section of the persona."""
    section_type: SectionType
    title: str
    fields: Dict[str, Any]
    is_approved: bool = False
    confidence_score: float = 0.0
    edit_count: int = 0
    last_edited: Optional[datetime] = None


@dataclass
class PersonaForReview:
    """Full persona prepared for review."""
    review_id: str
    user_id: str
    sections: Dict[SectionType, PersonaSection]
    overall_status: ApprovalStatus
    created_at: datetime
    updated_at: datetime
    approval_timestamp: Optional[datetime] = None
    edit_history: List[SectionEdit] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PersonaApproval:
    """
    Manages interactive persona review and approval.

    Allows users to review generated personas section by section,
    make edits, request regeneration, and give final approval.
    """

    # Section definitions with their fields
    SECTION_DEFINITIONS = {
        SectionType.IDENTITY: {
            "title": "Your Profile Identity",
            "fields": ["name", "archetype", "designation"],
            "editable": True,
            "description": "How you'll be presented to potential matches."
        },
        SectionType.PROFILE: {
            "title": "Profile Overview",
            "fields": ["profile_essence", "focus", "experience"],
            "editable": True,
            "description": "A summary of your background and focus areas."
        },
        SectionType.PHILOSOPHY: {
            "title": "Investment Philosophy",
            "fields": ["investment_philosophy"],
            "editable": True,
            "description": "Your approach to investments and partnerships."
        },
        SectionType.PREFERENCES: {
            "title": "What You're Looking For",
            "fields": ["what_theyre_looking_for"],
            "editable": True,
            "description": "The kind of opportunities and people you want to meet."
        },
        SectionType.STYLE: {
            "title": "Engagement Style",
            "fields": ["engagement_style"],
            "editable": True,
            "description": "How you prefer to work with matches."
        },
        SectionType.REQUIREMENTS: {
            "title": "Your Requirements",
            "fields": ["requirements_text"],
            "editable": True,
            "description": "What you're looking for in potential matches."
        },
        SectionType.OFFERINGS: {
            "title": "What You Offer",
            "fields": ["offerings_text"],
            "editable": True,
            "description": "What you bring to potential partnerships."
        }
    }

    def __init__(self):
        # Store personas under review (in production, use DB)
        self._reviews: Dict[str, PersonaForReview] = {}

        # Configuration
        self.require_all_sections_approved = os.getenv(
            "REQUIRE_ALL_SECTIONS_APPROVED", "false"
        ).lower() == "true"
        self.max_edit_history = int(os.getenv("MAX_PERSONA_EDIT_HISTORY", "50"))

    def create_review(
        self,
        user_id: str,
        persona_data: Dict[str, Any],
        requirements_text: str,
        offerings_text: str,
        confidence_scores: Optional[Dict[str, float]] = None
    ) -> PersonaForReview:
        """
        Create a persona review from generated data.

        Args:
            user_id: User identifier
            persona_data: Generated persona dictionary
            requirements_text: Requirements summary
            offerings_text: Offerings summary
            confidence_scores: Optional confidence per field

        Returns:
            PersonaForReview ready for user review
        """
        review_id = str(uuid4())
        now = datetime.utcnow()
        confidence_scores = confidence_scores or {}

        # Build sections from persona data
        sections = {}

        for section_type, section_def in self.SECTION_DEFINITIONS.items():
            fields = {}

            # Extract fields for this section
            for field_name in section_def["fields"]:
                if field_name == "requirements_text":
                    fields[field_name] = requirements_text
                elif field_name == "offerings_text":
                    fields[field_name] = offerings_text
                elif field_name in persona_data:
                    fields[field_name] = persona_data[field_name]

            # Calculate section confidence
            field_confidences = [
                confidence_scores.get(f, 0.8) for f in fields
            ]
            avg_confidence = (
                sum(field_confidences) / len(field_confidences)
                if field_confidences else 0.8
            )

            sections[section_type] = PersonaSection(
                section_type=section_type,
                title=section_def["title"],
                fields=fields,
                is_approved=False,
                confidence_score=avg_confidence,
                edit_count=0
            )

        review = PersonaForReview(
            review_id=review_id,
            user_id=user_id,
            sections=sections,
            overall_status=ApprovalStatus.DRAFT,
            created_at=now,
            updated_at=now,
            metadata={
                "original_persona": persona_data,
                "original_requirements": requirements_text,
                "original_offerings": offerings_text
            }
        )

        self._reviews[review_id] = review
        logger.info(f"Created persona review {review_id} for user {user_id}")

        return review

    def get_review(self, review_id: str) -> Optional[PersonaForReview]:
        """Get a persona review by ID."""
        return self._reviews.get(review_id)

    def get_review_for_user(self, user_id: str) -> Optional[PersonaForReview]:
        """Get the most recent review for a user."""
        user_reviews = [
            r for r in self._reviews.values()
            if r.user_id == user_id
        ]
        if not user_reviews:
            return None
        return max(user_reviews, key=lambda r: r.updated_at)

    def start_review(self, review_id: str) -> bool:
        """Mark review as in progress."""
        review = self._reviews.get(review_id)
        if not review:
            return False

        review.overall_status = ApprovalStatus.IN_REVIEW
        review.updated_at = datetime.utcnow()
        return True

    def get_section_for_review(
        self,
        review_id: str,
        section_type: SectionType
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific section prepared for review.

        Args:
            review_id: Review identifier
            section_type: Section to retrieve

        Returns:
            Section data with metadata for review UI
        """
        review = self._reviews.get(review_id)
        if not review or section_type not in review.sections:
            return None

        section = review.sections[section_type]
        section_def = self.SECTION_DEFINITIONS[section_type]

        return {
            "section_type": section_type.value,
            "title": section.title,
            "description": section_def["description"],
            "fields": section.fields,
            "is_approved": section.is_approved,
            "confidence_score": section.confidence_score,
            "edit_count": section.edit_count,
            "editable": section_def["editable"],
            "last_edited": section.last_edited.isoformat() if section.last_edited else None
        }

    def get_all_sections_summary(self, review_id: str) -> List[Dict[str, Any]]:
        """
        Get summary of all sections for overview.

        Args:
            review_id: Review identifier

        Returns:
            List of section summaries
        """
        review = self._reviews.get(review_id)
        if not review:
            return []

        summaries = []
        for section_type, section in review.sections.items():
            section_def = self.SECTION_DEFINITIONS[section_type]

            # Generate preview text
            preview = ""
            for field_name, value in section.fields.items():
                if value:
                    preview = str(value)[:100]
                    break

            summaries.append({
                "section_type": section_type.value,
                "title": section.title,
                "is_approved": section.is_approved,
                "confidence_score": section.confidence_score,
                "edit_count": section.edit_count,
                "preview": preview + "..." if len(preview) == 100 else preview,
                "needs_attention": section.confidence_score < 0.7
            })

        return summaries

    def edit_section_field(
        self,
        review_id: str,
        section_type: SectionType,
        field_name: str,
        new_value: Any,
        edit_source: EditSource = EditSource.USER,
        reason: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Edit a specific field in a section.

        Args:
            review_id: Review identifier
            section_type: Section to edit
            field_name: Field within section
            new_value: New value for the field
            edit_source: Source of the edit
            reason: Optional reason for the edit

        Returns:
            Tuple of (success, result)
        """
        review = self._reviews.get(review_id)
        if not review:
            return False, {"error": "Review not found"}

        if section_type not in review.sections:
            return False, {"error": "Section not found"}

        section = review.sections[section_type]
        if field_name not in section.fields:
            return False, {"error": "Field not found in section"}

        # Record edit
        old_value = section.fields[field_name]
        edit = SectionEdit(
            edit_id=str(uuid4()),
            section=section_type,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            edit_source=edit_source,
            timestamp=datetime.utcnow(),
            reason=reason
        )

        # Apply edit
        section.fields[field_name] = new_value
        section.edit_count += 1
        section.last_edited = datetime.utcnow()
        section.is_approved = False  # Reset approval on edit

        # Update review
        review.edit_history.append(edit)
        if len(review.edit_history) > self.max_edit_history:
            review.edit_history = review.edit_history[-self.max_edit_history:]

        review.overall_status = ApprovalStatus.PENDING_CHANGES
        review.updated_at = datetime.utcnow()

        logger.info(
            f"Edited {section_type.value}.{field_name} in review {review_id}"
        )

        return True, {
            "section": section_type.value,
            "field": field_name,
            "old_value": old_value,
            "new_value": new_value,
            "edit_count": section.edit_count
        }

    def approve_section(
        self,
        review_id: str,
        section_type: SectionType
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Approve a specific section.

        Args:
            review_id: Review identifier
            section_type: Section to approve

        Returns:
            Tuple of (success, result)
        """
        review = self._reviews.get(review_id)
        if not review:
            return False, {"error": "Review not found"}

        if section_type not in review.sections:
            return False, {"error": "Section not found"}

        review.sections[section_type].is_approved = True
        review.updated_at = datetime.utcnow()

        # Check if all sections approved
        all_approved = all(s.is_approved for s in review.sections.values())

        logger.info(f"Approved section {section_type.value} in review {review_id}")

        return True, {
            "section": section_type.value,
            "all_sections_approved": all_approved
        }

    def approve_all(self, review_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Approve all sections and finalize the persona.

        Args:
            review_id: Review identifier

        Returns:
            Tuple of (success, result with final persona)
        """
        review = self._reviews.get(review_id)
        if not review:
            return False, {"error": "Review not found"}

        # Mark all sections approved
        for section in review.sections.values():
            section.is_approved = True

        # Finalize
        review.overall_status = ApprovalStatus.APPROVED
        review.approval_timestamp = datetime.utcnow()
        review.updated_at = datetime.utcnow()

        # Build final persona
        final_persona = self._build_final_persona(review)

        logger.info(f"Approved all sections in review {review_id}")

        return True, {
            "review_id": review_id,
            "status": ApprovalStatus.APPROVED.value,
            "approval_timestamp": review.approval_timestamp.isoformat(),
            "final_persona": final_persona
        }

    def reject_review(
        self,
        review_id: str,
        reason: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Reject the entire persona and request regeneration.

        Args:
            review_id: Review identifier
            reason: Reason for rejection

        Returns:
            Tuple of (success, result)
        """
        review = self._reviews.get(review_id)
        if not review:
            return False, {"error": "Review not found"}

        review.overall_status = ApprovalStatus.REJECTED
        review.updated_at = datetime.utcnow()
        review.metadata["rejection_reason"] = reason

        logger.info(f"Rejected review {review_id}: {reason}")

        return True, {
            "review_id": review_id,
            "status": ApprovalStatus.REJECTED.value,
            "reason": reason,
            "can_regenerate": True
        }

    def _build_final_persona(self, review: PersonaForReview) -> Dict[str, Any]:
        """Build the final persona from approved sections."""
        final = {
            "user_id": review.user_id,
            "approved_at": review.approval_timestamp.isoformat()
            if review.approval_timestamp else None,
            "total_edits": sum(s.edit_count for s in review.sections.values())
        }

        # Collect all fields from sections
        for section in review.sections.values():
            for field_name, value in section.fields.items():
                # Special handling for requirements/offerings
                if field_name == "requirements_text":
                    final["requirements"] = value
                elif field_name == "offerings_text":
                    final["offerings"] = value
                else:
                    final[field_name] = value

        return final

    def get_review_progress(self, review_id: str) -> Dict[str, Any]:
        """
        Get review progress summary.

        Args:
            review_id: Review identifier

        Returns:
            Progress summary
        """
        review = self._reviews.get(review_id)
        if not review:
            return {}

        total_sections = len(review.sections)
        approved_sections = sum(
            1 for s in review.sections.values() if s.is_approved
        )
        total_edits = sum(s.edit_count for s in review.sections.values())
        low_confidence = [
            s.section_type.value for s in review.sections.values()
            if s.confidence_score < 0.7
        ]

        return {
            "review_id": review_id,
            "status": review.overall_status.value,
            "progress_percent": (approved_sections / total_sections * 100)
            if total_sections > 0 else 0,
            "sections_approved": approved_sections,
            "total_sections": total_sections,
            "total_edits": total_edits,
            "low_confidence_sections": low_confidence,
            "ready_to_approve": approved_sections == total_sections
            or not self.require_all_sections_approved
        }

    def get_edit_history(
        self,
        review_id: str,
        section_type: Optional[SectionType] = None
    ) -> List[Dict[str, Any]]:
        """
        Get edit history for the review.

        Args:
            review_id: Review identifier
            section_type: Optional filter by section

        Returns:
            List of edit records
        """
        review = self._reviews.get(review_id)
        if not review:
            return []

        history = review.edit_history
        if section_type:
            history = [e for e in history if e.section == section_type]

        return [
            {
                "edit_id": e.edit_id,
                "section": e.section.value,
                "field": e.field_name,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "source": e.edit_source.value,
                "timestamp": e.timestamp.isoformat(),
                "reason": e.reason
            }
            for e in history
        ]

    def request_regeneration(
        self,
        review_id: str,
        section_type: SectionType,
        guidance: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Request AI regeneration of a section.

        Args:
            review_id: Review identifier
            section_type: Section to regenerate
            guidance: Optional guidance for regeneration

        Returns:
            Request details (actual regeneration handled by persona_service)
        """
        review = self._reviews.get(review_id)
        if not review:
            return {"error": "Review not found"}

        return {
            "review_id": review_id,
            "section": section_type.value,
            "user_id": review.user_id,
            "current_fields": review.sections[section_type].fields,
            "guidance": guidance,
            "action": "regenerate_section"
        }


# Global instance
persona_approval = PersonaApproval()
