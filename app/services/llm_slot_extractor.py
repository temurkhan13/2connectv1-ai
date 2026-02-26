"""
LLM-based Slot Extraction Service.

Uses OpenAI/Claude to extract structured slot data from freeform user text.
This replaces regex-based extraction with true language understanding.

Key advantages over regex:
1. Understands context ("looking for investors" = founder, not investor)
2. Handles typos, unusual phrasing, and corrections naturally
3. Extracts multiple slots from a single response
4. Generates contextual follow-up questions
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMExtractedSlot:
    """A slot extracted by the LLM."""
    name: str
    value: Any
    confidence: float
    reasoning: str  # Why the LLM extracted this value


@dataclass
class LLMExtractionResult:
    """Result of LLM slot extraction."""
    extracted_slots: Dict[str, LLMExtractedSlot]
    user_type_inference: str  # "founder", "investor", "advisor", etc.
    follow_up_question: str  # Contextual next question
    missing_slots: List[str]  # What's still needed
    understanding_summary: str  # Brief summary of what LLM understood


# Slot definitions for the LLM prompt
SLOT_DEFINITIONS = {
    "user_type": {
        "description": "The user's primary role in the startup ecosystem",
        "options": ["Founder/Entrepreneur", "Angel Investor", "VC Partner", "Corporate Executive", "Mentor/Advisor", "Service Provider"],
        "extraction_hint": "IMPORTANT: If user mentions 'looking for investors', 'seeking funding', 'raising a round', they are a FOUNDER seeking investors, NOT an investor themselves. Only classify as investor if they explicitly say they INVEST money in startups."
    },
    "primary_goal": {
        "description": "What the user wants to achieve on the platform",
        "options": ["Raise Funding", "Find Co-founder", "Seek Mentorship", "Explore Partnerships", "Invest in Startups", "Offer Services"],
        "extraction_hint": "Match their stated intention to one of these goals"
    },
    "industry_focus": {
        "description": "Industries or sectors the user focuses on",
        "type": "multi_select",
        "options": ["Technology/SaaS", "Healthcare/Biotech", "FinTech", "E-commerce", "AI/ML", "CleanTech", "EdTech", "Consumer", "Enterprise", "Other"],
        "extraction_hint": "Can select multiple. Infer from company description or stated interests."
    },
    "stage_preference": {
        "description": "Company stage they work with or are at",
        "options": ["Pre-seed", "Seed", "Series A", "Series B+", "Growth", "Any Stage"],
        "extraction_hint": "For founders: their current stage. For investors: stages they invest in."
    },
    "funding_need": {
        "description": "Amount of funding sought (for founders)",
        "type": "text",
        "extraction_hint": "Extract specific amounts like '$2M', '$500K-$1M', etc."
    },
    "check_size": {
        "description": "Typical investment amount (for investors)",
        "type": "text",
        "extraction_hint": "Extract investment ranges like '$25K-$100K', '$1M-$5M', etc."
    },
    "geographic_focus": {
        "description": "Geographic regions of interest",
        "type": "multi_select",
        "options": ["US", "UK", "Europe", "Asia", "MENA", "Global", "Other"],
        "extraction_hint": "Extract mentioned regions or countries"
    },
    "company_name": {
        "description": "Name of the user's company (if founder/executive)",
        "type": "text",
        "extraction_hint": "Extract company name if mentioned"
    },
    "role_title": {
        "description": "User's job title or role",
        "type": "text",
        "extraction_hint": "Extract titles like 'CEO', 'CTO', 'Partner', etc."
    },
    "experience_years": {
        "description": "Years of relevant experience",
        "type": "text",
        "extraction_hint": "Extract mentioned experience like '10+ years', '5 years in tech', etc."
    },
    "offerings": {
        "description": "What the user can offer to connections",
        "type": "text",
        "extraction_hint": "Extract what they say they can provide: capital, mentorship, introductions, expertise, etc."
    },
    "requirements": {
        "description": "What the user needs from connections",
        "type": "text",
        "extraction_hint": "Extract what they're looking for: funding, advisors, talent, partnerships, etc."
    },
    "timeline": {
        "description": "Timeline for their goals",
        "type": "text",
        "extraction_hint": "Extract timeframes like 'next 3 months', 'Q2 2024', 'this year', etc."
    },
    "name": {
        "description": "User's name",
        "type": "text",
        "extraction_hint": "Extract their name if they introduce themselves"
    },
    # COFOUNDER-specific slots - MUST match SlotSchema for progress tracking
    "skills_have": {
        "description": "Skills the user brings to a co-founder partnership",
        "type": "multi_select",
        "options": ["Technical/Engineering", "Product Management", "Sales/Business Development", "Marketing/Growth", "Finance/Operations", "Design/UX", "Domain Expertise", "Fundraising Experience"],
        "extraction_hint": "Map their stated skills: 'backend developer' → Technical/Engineering, 'sold products' → Sales/Business Development"
    },
    "skills_need": {
        "description": "Skills the user needs in a co-founder",
        "type": "multi_select",
        "options": ["Technical/Engineering", "Product Management", "Sales/Business Development", "Marketing/Growth", "Finance/Operations", "Design/UX", "Domain Expertise", "Fundraising Experience"],
        "extraction_hint": "Map their needs: 'frontend developer' → Technical/Engineering, 'sales experience' → Sales/Business Development, 'fundraising' → Fundraising Experience"
    },
    "commitment_level": {
        "description": "Expected commitment level for co-founder",
        "options": ["Full-time immediately", "Full-time after funding", "Part-time initially", "Nights & weekends", "Flexible/discuss"],
        "extraction_hint": "Extract commitment expectations from phrases like 'work together daily', 'full-time', 'weekends'"
    },
    "equity_expectations": {
        "description": "Equity split expectations for co-founder",
        "options": ["Equal split (50/50)", "Majority for existing founder", "Based on contribution", "Open to discuss", "Vesting with cliff"],
        "extraction_hint": "Extract from mentions of '60/40 split', 'equal equity', 'vesting'"
    },
    # INVESTOR-specific slots - MUST match SlotSchema for progress tracking
    "portfolio_size": {
        "description": "Number of current investments in portfolio",
        "type": "number",
        "extraction_hint": "Extract numbers: '20 companies', 'dozen investments', 'portfolio of 50'"
    },
    "investment_thesis": {
        "description": "Investment philosophy and focus areas",
        "type": "text",
        "extraction_hint": "Extract their investment approach, what they look for, why they invest in certain companies"
    },
    # FOUNDER-specific slots - MUST match SlotSchema for progress tracking
    "company_stage": {
        "description": "Current stage of the startup",
        "options": ["Idea", "MVP", "Product-Market Fit", "Scaling", "Established"],
        "extraction_hint": "Map stages: 'just an idea' → Idea, 'working prototype' → MVP, 'customers paying' → Product-Market Fit"
    },
    "team_size": {
        "description": "Current team size",
        "type": "number",
        "extraction_hint": "Extract numbers: 'team of 5', 'solo founder', '12 employees'"
    },
    # HIRING-specific slots - MUST match SlotSchema for progress tracking
    "role_type": {
        "description": "Type of role being hired for",
        "options": ["Engineering/Technical", "Product Management", "Sales/Business Development", "Marketing/Growth", "Operations", "Executive/C-Suite", "Finance/Legal", "Other"],
        "extraction_hint": "Map role types: 'developer' → Engineering/Technical, 'product manager' → Product Management, 'sales' → Sales/Business Development"
    },
    "seniority_level": {
        "description": "Seniority level for the role",
        "options": ["Junior/Entry-level", "Mid-level", "Senior", "Lead/Staff", "Director", "VP/Executive", "C-Suite"],
        "extraction_hint": "Extract seniority: 'junior dev' → Junior/Entry-level, 'senior engineer' → Senior, 'CTO' → C-Suite"
    },
    "remote_preference": {
        "description": "Work location preference",
        "options": ["Fully Remote", "Hybrid", "On-site Only", "Flexible"],
        "extraction_hint": "Extract work preferences: 'remote-first' → Fully Remote, 'in office' → On-site Only"
    },
    "compensation_range": {
        "description": "Budget or compensation expectations for the role",
        "type": "range",
        "extraction_hint": "Extract salary ranges: '$100K-$150K', '£80K', 'competitive package'"
    },
    # MENTORSHIP-specific slots - MUST match SlotSchema for progress tracking
    "mentorship_areas": {
        "description": "Areas where mentorship is sought or offered",
        "type": "multi_select",
        "options": ["Leadership & Management", "Technical/Engineering", "Go-to-Market Strategy", "Fundraising", "Hiring & Team Building", "Product Development", "Sales & Business Development", "Marketing & Growth", "Operations & Scaling", "Career Development"],
        "extraction_hint": "Map mentorship areas from their interests or expertise"
    },
    "mentorship_format": {
        "description": "Preferred mentorship format",
        "options": ["Weekly calls", "Bi-weekly calls", "Monthly sessions", "Async messaging", "Ad-hoc as needed"],
        "extraction_hint": "Extract from meeting preferences: 'regular calls' → Weekly calls, 'flexible' → Ad-hoc as needed"
    },
    "mentorship_commitment": {
        "description": "Hours per month available for mentorship",
        "options": ["1-2 hours/month", "3-5 hours/month", "5-10 hours/month", "10+ hours/month", "Flexible"],
        "extraction_hint": "Extract time commitment: 'few hours a month' → 3-5 hours/month"
    }
}


class LLMSlotExtractor:
    """
    Extracts slot values from user text using LLM comprehension.

    Unlike regex-based extraction, this:
    - Understands context and intent
    - Handles corrections naturally
    - Extracts multiple slots from freeform text
    - Generates contextual follow-up questions
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Use gpt-4o-mini for faster responses (3-5x faster than gpt-4o)
        # Still excellent for slot extraction. Override with OPENAI_EXTRACTION_MODEL env var.
        # Note: gpt-4o-mini has 128K context, sufficient for our prompts.
        self.model = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-4o-mini")

    def extract_slots(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        already_filled_slots: Optional[Dict[str, Any]] = None,
        target_slots: Optional[List[str]] = None
    ) -> LLMExtractionResult:
        """
        Extract slot values from user message using LLM comprehension.

        Args:
            user_message: The user's latest message
            conversation_history: Previous turns for context
            already_filled_slots: Slots already extracted in this session
            target_slots: Specific slots to focus on (None = all)

        Returns:
            LLMExtractionResult with extracted slots and follow-up
        """
        already_filled = already_filled_slots or {}
        history = conversation_history or []

        # Build the extraction prompt
        system_prompt = self._build_system_prompt(already_filled, target_slots)

        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history for context
        for turn in history[-6:]:  # Last 6 turns for context
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", "")
            })

        # Add current message
        messages.append({
            "role": "user",
            "content": f"Extract information from this message:\n\n\"{user_message}\""
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1500
            )

            result_text = response.choices[0].message.content
            result_data = json.loads(result_text)

            return self._parse_llm_response(result_data, already_filled)

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Return empty result on failure
            return LLMExtractionResult(
                extracted_slots={},
                user_type_inference="unknown",
                follow_up_question="Could you tell me more about yourself and what you're looking for?",
                missing_slots=list(SLOT_DEFINITIONS.keys()),
                understanding_summary="I had trouble understanding your message. Could you rephrase?"
            )

    def _build_system_prompt(
        self,
        already_filled: Dict[str, Any],
        target_slots: Optional[List[str]]
    ) -> str:
        """Build the system prompt for extraction."""

        # Filter to target slots if specified
        slots_to_extract = SLOT_DEFINITIONS
        if target_slots:
            slots_to_extract = {k: v for k, v in SLOT_DEFINITIONS.items() if k in target_slots}

        # Remove already filled slots from extraction targets
        remaining_slots = {k: v for k, v in slots_to_extract.items() if k not in already_filled}

        slot_descriptions = []
        for name, definition in remaining_slots.items():
            desc = f"- {name}: {definition['description']}"
            if 'options' in definition:
                desc += f"\n  Options: {', '.join(definition['options'])}"
            if 'extraction_hint' in definition:
                desc += f"\n  Hint: {definition['extraction_hint']}"
            slot_descriptions.append(desc)

        already_filled_text = ""
        if already_filled:
            filled_items = [f"- {k}: {v}" for k, v in already_filled.items()]
            already_filled_text = f"""
## ALREADY COLLECTED - DO NOT ASK AGAIN
The following information has ALREADY been collected. Do NOT ask for these again:
{chr(10).join(filled_items)}

CRITICAL: Your follow_up_question must NEVER ask for information listed above. Only ask about what's MISSING.
"""

        return f"""You are a warm, curious conversationalist helping someone tell their story.

## Your Task
1. Extract new profile information from the user's latest message
2. Generate a natural, conversational follow-up that BUILDS ON what they just shared

## ⚠️ CRITICAL: True Indirect Elicitation

You MUST sound like a curious friend at a coffee chat, NOT a survey. The key is to pick up on SPECIFIC details they mentioned.

### THE GOLDEN RULE
Your follow-up must reference something SPECIFIC from their last message. Don't ask generic questions - react to THEIR story.

### EXAMPLES OF PICKING UP ON SPECIFICS

User says: "I'm building an AI platform for medical diagnosis"
❌ BAD: "What excites you most about healthtech?" (generic)
✅ GOOD: "Medical diagnosis is fascinating - are you focusing on a specific condition or specialty?"

User says: "I have a prototype ready and need business help"
❌ BAD: "How are you thinking about funding?" (doesn't build on prototype)
✅ GOOD: "A working prototype is huge! What's the biggest gap you're trying to fill on the business side?"

User says: "I offer technical expertise in AI and I'm looking for business partners"
❌ BAD: "What kind of relationship would be valuable?" (generic)
✅ GOOD: "With your AI background, are you thinking more co-founder or advisor-type partnerships?"

### VARY YOUR QUESTION PATTERNS
DON'T keep using the same patterns. Rotate through these styles:
- Hook on a specific detail: "You mentioned [X] - tell me more about that"
- Curious follow-up: "That's interesting - how did you get into [specific thing they said]?"
- Dig deeper: "When you say [their words], what does that look like for you?"
- Natural progression: "So with [thing they mentioned], what's the next step you're focused on?"
- Challenge/explore: "That's ambitious - what's the hardest part of [their goal]?"

### NEVER USE THESE REPEATEDLY
Avoid using the same pattern twice in a conversation:
- "What excites you most about X?" (use once max)
- "How are you thinking about X?" (use once max)
- "Tell me more about X" (use once max)

### STAY INDIRECT (default mode)
- Follow the user's natural conversational thread
- Infer information from context rather than asking directly
- Match their energy - if they're enthusiastic, be enthusiastic back

### ONLY GO DIRECT when:
- User gives extremely terse responses (single words)
- You've tried indirect 2+ times with no result
- User explicitly asks "what do you need from me?"

## Critical Rules
1. NEVER repeat back what the user just told you (no "You mentioned...", "I see you're...", "Thanks for sharing...")
2. NEVER ask for information that's already been collected (see ALREADY COLLECTED section)
3. Keep responses SHORT - one conversational question, max 15 words
4. Pick up on SPECIFIC words/phrases from their message - don't ask generic questions
5. If user says "looking for investors" or "raising funding" - they are a FOUNDER, not an investor
6. CEO, Founder, Co-founder = Founder/Entrepreneur
7. Match their energy - enthusiastic users get excited responses, thoughtful users get thoughtful questions
8. Check conversation history - DON'T repeat question patterns you already used

## Slots to Extract
{chr(10).join(slot_descriptions)}
{already_filled_text}
## Response Format
Return valid JSON:
{{
    "extracted_slots": {{
        "slot_name": {{
            "value": "extracted value",
            "confidence": 0.0-1.0,
            "reasoning": "why you extracted this"
        }}
    }},
    "user_type_inference": "founder|investor|advisor|executive|service_provider|unknown",
    "understanding_summary": "INTERNAL ONLY - not shown to user - your private notes",
    "missing_important_slots": ["slots", "still", "needed"],
    "follow_up_question": "QUESTION ONLY - no acknowledgment, no summary, no 'The user is...'"
}}

## CRITICAL: follow_up_question Rules
The follow_up_question field must be:
- A SHORT (max 15 words), conversational question
- Indirect and curious-sounding, not form-like
- ONLY a question - no acknowledgments, no summaries

MUST NOT contain:
- "The user is..." or "They are..."
- "Nice to meet you!" or "Thanks for sharing!" or "Great!"
- "What is your X?" (sounds like a form)
- Any repetition of what they said

WRONG: "The user is a fintech founder. Nice to meet you! What's your company name?"
WRONG: "What is your investment check size?"
WRONG: "Thanks for sharing! What industry are you focused on?"

RIGHT: "How are you thinking about check sizes?"
RIGHT: "What space gets you most excited?"
RIGHT: "Where on this journey is your company?"

## Good vs Bad Follow-up Questions

BAD (generic, doesn't pick up on specifics):
User said: "I'm passionate about using AI for early disease detection"
Response: "What excites you most about healthtech?" ← Ignores their specific interest!

GOOD (picks up on their specific words):
User said: "I'm passionate about using AI for early disease detection"
Response: "Early detection is powerful - are you focusing on any particular diseases?"

BAD (repeats pattern already used):
If you already asked "How are you thinking about X?", DON'T ask it again.

GOOD (varies the pattern):
First question: "How are you thinking about funding?"
Next question: "With your background in [X], what's the biggest challenge right now?"

BAD (generic question that ignores context):
User said: "I have experience with startups that reached Series A"
Response: "What stage companies do you work with?"

GOOD (builds on their experience):
User said: "I have experience with startups that reached Series A"
Response: "Getting to Series A is no small feat - what made those companies successful?"

## Example

User already provided: name=Sarah, company=TechVenture, stage=Series A
User says: "We're looking to raise $3M"

Response:
{{
    "extracted_slots": {{
        "funding_need": {{"value": "$3M", "confidence": 0.95, "reasoning": "Explicitly stated raise amount"}}
    }},
    "user_type_inference": "founder",
    "understanding_summary": "Founder raising $3M Series A",
    "missing_important_slots": ["timeline", "geographic_focus"],
    "follow_up_question": "How are you thinking about timing?"
}}

Extract only what's NEW in this message. Generate a SHORT follow-up about what's STILL MISSING."""

    def _parse_llm_response(
        self,
        response_data: Dict[str, Any],
        already_filled: Dict[str, Any]
    ) -> LLMExtractionResult:
        """Parse LLM response into structured result."""

        extracted_slots = {}
        raw_slots = response_data.get("extracted_slots", {})

        for slot_name, slot_data in raw_slots.items():
            if isinstance(slot_data, dict):
                extracted_slots[slot_name] = LLMExtractedSlot(
                    name=slot_name,
                    value=slot_data.get("value"),
                    confidence=float(slot_data.get("confidence", 0.8)),
                    reasoning=slot_data.get("reasoning", "")
                )
            else:
                # Handle simple value format
                extracted_slots[slot_name] = LLMExtractedSlot(
                    name=slot_name,
                    value=slot_data,
                    confidence=0.8,
                    reasoning=""
                )

        # Clean follow-up question - remove third-person summaries
        raw_follow_up = response_data.get("follow_up_question", "")
        clean_follow_up = self._clean_follow_up_question(raw_follow_up)

        return LLMExtractionResult(
            extracted_slots=extracted_slots,
            user_type_inference=response_data.get("user_type_inference", "unknown"),
            follow_up_question=clean_follow_up,
            missing_slots=response_data.get("missing_important_slots", []),
            understanding_summary=response_data.get("understanding_summary", "")
        )

    def _clean_follow_up_question(self, text: str) -> str:
        """
        Remove third-person summaries from follow-up questions.

        LLMs sometimes prefix questions with summaries like:
        "The user is a founder. What's your timeline?"

        This strips those prefixes to keep responses clean.
        """
        import re

        if not text:
            return text

        # Patterns that indicate third-person summary (to remove)
        remove_patterns = [
            r"^The user is[^.!?]*[.!?]\s*",  # "The user is a founder."
            r"^They are[^.!?]*[.!?]\s*",  # "They are raising funding."
            r"^This user[^.!?]*[.!?]\s*",  # "This user wants..."
            r"^The founder[^.!?]*[.!?]\s*",  # "The founder is..."
            r"^[A-Z][a-z]+ is a[^.!?]*[.!?]\s*",  # "Sarah is a founder."
            r"^Based on[^.!?]*[.!?]\s*",  # "Based on what you said..."
            r"^From what you[^.!?]*[.!?]\s*",  # "From what you mentioned..."
            r"^I understand[^.!?]*[.!?]\s*",  # "I understand you're a founder."
            r"^It sounds like[^.!?]*[.!?]\s*",  # "It sounds like you're..."
            r"^So you[^.!?]*[.!?]\s*",  # "So you're raising..."
            r"^You mentioned[^.!?]*[.!?]\s*",  # "You mentioned being..."
            r"^You're a[^.!?]*[.!?]\s*",  # "You're a fintech founder."
            r"^I see that[^.!?]*[.!?]\s*",  # "I see that you..."
            r"^Got it[^.!?]*[.!?]\s*",  # "Got it, you're a founder."
        ]

        cleaned = text
        for pattern in remove_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # If we stripped everything, try to salvage by taking last sentence
        if not cleaned.strip():
            sentences = text.split('. ')
            if len(sentences) > 1:
                # Check if first sentence is a summary
                first_lower = sentences[0].lower()
                if any(word in first_lower for word in ['the user', 'they are', 'this user', 'founder is', 'you are a', "you're a"]):
                    cleaned = '. '.join(sentences[1:])
                else:
                    cleaned = text
            else:
                cleaned = text

        return cleaned.strip()

    def generate_response(
        self,
        extraction_result: LLMExtractionResult,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a natural, contextual response based on extraction.

        This creates conversational responses that:
        - Acknowledge what was understood
        - Reference previous context naturally
        - Ask for missing information conversationally
        """

        # If LLM already provided a good follow-up, use it directly
        # NEVER concatenate understanding_summary - it's internal only
        if extraction_result.follow_up_question:
            return extraction_result.follow_up_question

        # Fallback response generation
        if extraction_result.missing_slots:
            missing = extraction_result.missing_slots[:2]
            return f"Thanks for sharing! Could you tell me about your {' and '.join(missing)}?"

        return "I have all the information I need. Would you like to complete your profile?"


# Global instance
llm_slot_extractor = LLMSlotExtractor()
