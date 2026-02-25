"""
Templates Router - Expose use case templates to frontend.

Provides endpoints to list and retrieve objective-specific templates
for customizing AI behavior based on user goals.
"""
from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel

from app.services.use_case_templates import (
    list_objectives,
    get_template,
    ObjectiveType,
)

router = APIRouter(prefix="/templates", tags=["templates"])


class ObjectiveSummary(BaseModel):
    """Summary of an objective type."""
    code: str
    display_name: str
    description: str


class TemplateResponse(BaseModel):
    """Full template details for a specific objective."""
    code: str
    display_name: str
    description: str
    key_questions: List[str]
    success_criteria: List[str]


@router.get("/objectives", response_model=List[ObjectiveSummary])
async def get_objectives():
    """
    List all available objective types with summaries.

    Returns a list of 7 objective types:
    - fundraising: Seeking Investment
    - investing: Looking to Invest
    - hiring: Hiring Talent
    - partnership: Strategic Partnership
    - mentorship: Mentorship
    - cofounder: Finding Co-founder
    - networking: Professional Networking
    """
    objectives = list_objectives()
    return [
        ObjectiveSummary(
            code=obj['code'],
            display_name=obj['display_name'],
            description=obj['description']
        )
        for obj in objectives
    ]


@router.get("/objectives/{objective}", response_model=TemplateResponse)
async def get_objective_template(objective: str):
    """
    Get full template for a specific objective.

    Args:
        objective: One of: fundraising, investing, hiring, partnership,
                   mentorship, cofounder, networking

    Returns:
        Full template with key_questions and success_criteria
    """
    try:
        template = get_template(objective)
        # Handle both enum and string types for objective
        code = template.objective.value if hasattr(template.objective, 'value') else str(template.objective)
        return TemplateResponse(
            code=code,
            display_name=template.display_name,
            description=template.description,
            key_questions=template.key_questions,
            success_criteria=template.success_criteria
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Template not found: {objective}")
