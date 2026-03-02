"""
LLM-based Slot Extraction Service.

Uses Claude Sonnet 4.5 to extract structured slot data from freeform user text.
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
from anthropic import Anthropic

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
    is_off_topic: bool  # True if user asked off-topic/general knowledge question


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
    "geography": {
        "description": "Geographic regions of interest",
        "type": "multi_select",
        "options": ["UK", "US", "Europe", "Asia", "Middle East", "Latin America", "Africa", "Global/Remote"],
        "extraction_hint": "Extract mentioned regions or countries. Map specific countries to regions: London/UK → UK, Silicon Valley/US → US, etc."
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
        "description": "Expected commitment level for CO-FOUNDER PARTNERSHIP (NOT their current job status)",
        "options": ["Full-time immediately", "Full-time after funding", "Part-time initially", "Nights & weekends", "Flexible/discuss"],
        "extraction_hint": "CRITICAL: This is about commitment to the CO-FOUNDER SEARCH/STARTUP, NOT their current employment. If user mentions having a 'full-time job' or 'day job', that means they can only commit 'Part-time initially' or 'Nights & weekends' to the startup. Only extract 'Full-time immediately' if they explicitly say they're available full-time for the STARTUP/partnership right now (e.g., 'I quit my job', 'I'm full-time on this', 'ready to go all-in')."
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


# Semantic topic clusters - questions in same cluster are semantically equivalent
# If ANY keyword from a cluster was asked, ALL keywords in that cluster are "covered"
SEMANTIC_TOPIC_CLUSTERS = {
    "goals": ["goal", "goals", "objective", "objectives", "priority", "priorities",
              "aspiration", "aspirations", "ambition", "ambitions", "vision", "aim",
              "aims", "target", "targets", "hope", "hopes", "dream", "dreams",
              "what you want", "what do you want", "looking to achieve", "trying to achieve"],
    "challenges": ["challenge", "challenges", "obstacle", "obstacles", "blocker", "blockers",
                   "struggle", "struggles", "difficulty", "difficulties", "problem", "problems",
                   "pain point", "pain points", "issue", "issues", "barrier", "barriers",
                   "what's hard", "what's difficult", "keeps you up"],
    "skills": ["skill", "skills", "expertise", "experience", "strength", "strengths",
               "capability", "capabilities", "superpower", "superpowers", "good at",
               "excel at", "specialize", "specialty", "specialization", "background"],
    "needs": ["need", "needs", "requirement", "requirements", "support", "help",
              "gap", "gaps", "looking for", "seeking", "searching for", "want from",
              "need from", "require", "missing", "lack"],
    "offers": ["offer", "offers", "offering", "offerings", "provide", "bring",
               "contribute", "give", "share", "can do", "able to"],
    "geography": ["geography", "location", "region", "country", "where", "based",
                  "operate", "market", "markets", "uk", "us", "europe", "asia"],
    "stage": ["stage", "stages", "phase", "level", "round", "seed", "series",
              "pre-seed", "growth", "early-stage", "late-stage"],
    "industry": ["industry", "industries", "sector", "sectors", "space", "field",
                 "domain", "vertical", "market", "niche", "focus area"]
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
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Use Claude Sonnet 4.5 for warm, engaging conversations
        # Excellent at following detailed instruction prompts for personalized questions
        # Override with ANTHROPIC_EXTRACTION_MODEL env var.
        self.model = os.getenv("ANTHROPIC_EXTRACTION_MODEL", "claude-sonnet-4-5-20250929")

    def _detect_covered_topics(self, conversation_history: List[Dict[str, str]]) -> List[str]:
        """
        Analyze conversation history to find which semantic topics have been asked.

        Returns list of topic cluster names that have been covered.
        """
        covered = set()

        # Only look at assistant messages (questions that were asked)
        for turn in conversation_history or []:
            if turn.get("role") == "assistant":
                content = turn.get("content", "").lower()

                # Check each topic cluster
                for topic_name, keywords in SEMANTIC_TOPIC_CLUSTERS.items():
                    if any(kw in content for kw in keywords):
                        covered.add(topic_name)

        return list(covered)

    def _is_question_repetitive(self, question: str, covered_topics: List[str]) -> bool:
        """
        Check if a generated question covers an already-asked topic.

        Returns True if the question is semantically repetitive.
        """
        if not question or not covered_topics:
            return False

        question_lower = question.lower()

        for topic_name in covered_topics:
            keywords = SEMANTIC_TOPIC_CLUSTERS.get(topic_name, [])
            if any(kw in question_lower for kw in keywords):
                logger.warning(f"Question repeats covered topic '{topic_name}': {question[:50]}...")
                return True

        return False

    def _get_diversified_question(self, covered_topics: List[str], missing_slots: List[str]) -> str:
        """
        Generate a question about a topic that HASN'T been covered yet.

        Uses missing_slots to pick relevant uncovered topics.
        """
        # Map slots to topics
        slot_to_topic = {
            "primary_goal": "goals",
            "requirements": "needs",
            "offerings": "offers",
            "geography": "geography",
            "stage_preference": "stage",
            "industry_focus": "industry"
        }

        # Engaging questions for each topic (fallback if LLM keeps repeating)
        topic_questions = {
            "goals": "I'm curious — what does success look like for you in the next 12 months? What's the big milestone you're chasing?",
            "needs": "Every journey has its gaps — what kind of support or resources would really move the needle for you right now?",
            "offers": "What's your superpower? I'd love to hear what unique value you bring to the table when partnering with someone.",
            "geography": "I'm curious about your scope — which regions or markets are you most focused on or excited about?",
            "stage": "What stage companies do you typically work with? I imagine that shapes a lot about how you operate.",
            "industry": "What industries or sectors light you up? I find that passion often follows expertise.",
            "challenges": "What's the biggest obstacle standing between you and your next milestone? Sometimes naming it helps.",
            "skills": "I'd love to hear your story — what's your background, and how did it lead you to where you are now?"
        }

        # Find a topic that's NOT covered and IS relevant to missing slots
        for slot in missing_slots:
            topic = slot_to_topic.get(slot)
            if topic and topic not in covered_topics:
                return topic_questions.get(topic, "")

        # Fallback: any uncovered topic
        all_topics = set(SEMANTIC_TOPIC_CLUSTERS.keys())
        uncovered = all_topics - set(covered_topics)

        if uncovered:
            topic = list(uncovered)[0]
            return topic_questions.get(topic, "Tell me more about yourself and what you're looking for.")

        # All topics covered - signal completion
        return "I think I have a good picture now. Is there anything else you'd like to add?"

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

        # BUG-001 FIX: Check for completion intent BEFORE calling LLM
        # If user signals they're done, return early with empty follow_up_question
        if self._user_wants_to_finish(user_message, already_filled):
            logger.info(f"User completion signal detected in message: '{user_message[:50]}...'")
            return LLMExtractionResult(
                extracted_slots={},
                user_type_inference=already_filled.get("user_type", "unknown"),
                follow_up_question="",  # Empty = no more questions
                missing_slots=[],
                understanding_summary="User signaled completion",
                is_off_topic=False
            )

        # BUG-002 FIX: Detect which semantic topics have already been asked
        # This prevents GPT-4o-mini from asking "goals" vs "objectives" vs "priorities"
        covered_topics = self._detect_covered_topics(history)
        if covered_topics:
            logger.info(f"Topics already covered in conversation: {covered_topics}")

        # Build the extraction prompt with covered topics
        system_prompt = self._build_system_prompt(already_filled, target_slots, covered_topics)

        # Build conversation context (Anthropic API: system is separate, messages are user/assistant only)
        messages = []

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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=system_prompt,
                messages=messages,
                temperature=0.1  # Low temperature for consistent extraction
            )

            result_text = response.content[0].text
            result_data = json.loads(result_text)

            result = self._parse_llm_response(result_data, already_filled)

            # BUG-002 FIX: Check if LLM still generated a repetitive question
            # If so, auto-replace with a diversified question
            if result.follow_up_question and self._is_question_repetitive(result.follow_up_question, covered_topics):
                logger.warning(f"LLM generated repetitive question, auto-diversifying...")
                diversified = self._get_diversified_question(covered_topics, result.missing_slots)
                result = LLMExtractionResult(
                    extracted_slots=result.extracted_slots,
                    user_type_inference=result.user_type_inference,
                    follow_up_question=diversified,  # Replace with non-repetitive question
                    missing_slots=result.missing_slots,
                    understanding_summary=result.understanding_summary,
                    is_off_topic=result.is_off_topic
                )

            return result

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Return empty result on failure (still engaging)
            return LLMExtractionResult(
                extracted_slots={},
                user_type_inference="unknown",
                follow_up_question="I'd love to hear more about your journey — what brought you here today, and what are you hoping to find?",
                missing_slots=list(SLOT_DEFINITIONS.keys()),
                understanding_summary="I had trouble understanding your message. Could you rephrase?",
                is_off_topic=False
            )

    def _build_system_prompt(
        self,
        already_filled: Dict[str, Any],
        target_slots: Optional[List[str]],
        covered_topics: Optional[List[str]] = None
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

        # BUG-002 FIX: Explicit topic blocking based on conversation analysis
        covered_topics_text = ""
        if covered_topics:
            # Build explicit forbidden keywords list from covered topics
            forbidden_keywords = []
            for topic in covered_topics:
                keywords = SEMANTIC_TOPIC_CLUSTERS.get(topic, [])
                forbidden_keywords.extend(keywords[:5])  # Top 5 keywords per topic

            covered_topics_text = f"""
## 🚨 ABSOLUTELY FORBIDDEN TOPICS - ALREADY ASKED
The following topics have ALREADY been discussed. Your follow_up_question MUST NOT contain ANY of these words or concepts:

FORBIDDEN WORDS: {', '.join(forbidden_keywords[:20])}

COVERED TOPICS: {', '.join(covered_topics)}

If you ask about ANY of these topics, you will be penalized. Ask about something COMPLETELY DIFFERENT.
Choose from UNCOVERED topics like: {', '.join(set(SEMANTIC_TOPIC_CLUSTERS.keys()) - set(covered_topics))}
"""

        # Determine which REQUIRED slots are still missing
        required_slots = ["primary_goal", "requirements", "offerings", "user_type", "industry_focus", "stage_preference", "geography"]
        missing_required = [s for s in required_slots if s not in already_filled]

        missing_required_text = ""
        if missing_required:
            missing_required_text = f"""
## ⚠️ PRIORITY: REQUIRED SLOTS STILL MISSING
These slots are REQUIRED for profile completion. Your follow-up question should naturally guide toward collecting one of these:
{', '.join(missing_required)}

- requirements = What they NEED from connections (funding, advisors, partnerships, etc.)
- offerings = What they can OFFER to connections (capital, expertise, introductions, etc.)
- geography = Where they're focused (UK, US, Europe, Asia, Global, etc.)
- stage_preference = What company stages they work with (Pre-seed, Seed, Series A, etc.)

CRITICAL: If 3+ required slots are missing, prioritize collecting them over drilling into details.
"""

        return f"""You are a warm, genuinely curious interviewer who builds real rapport. Think of yourself as a trusted friend who's fascinated by people's journeys.

## Your Role
You make every person feel HEARD and VALUED. You're not filling out a form — you're having a real conversation with someone interesting. You reference SPECIFIC details they shared, show genuine enthusiasm for their unique story, and ask follow-ups that prove you were really listening.

## Primary Objectives
1. Make the user feel genuinely understood by MIRRORING BACK specific details they mentioned
2. Show authentic curiosity about their unique journey ("That's fascinating...", "I love that approach...")
3. Build rapport through warm, personalized acknowledgments before asking the next question
4. Extract information naturally through engaged conversation, not form-filling
5. NEVER repeat questions about topics already covered

## 🌟 ENGAGEMENT STYLE (CRITICAL)

**ROBOTIC (NEVER DO THIS):**
- "Thanks for sharing! What are your goals?"
- "Got it. What industry are you in?"
- "I see. What's your timeline?"

**ENGAGING (ALWAYS DO THIS):**
- "That's a fascinating journey — pivoting from sales into tech takes real vision. What sparked that shift for you?"
- "I love the consultative approach; it sounds like you're positioning as a strategic partner, not just another vendor. What's been the response so far?"
- "Building an AI tool for hospitals — that's such a complex space to crack. What's been the most surprising thing you've learned?"

## 🔑 PERSONALIZATION RULES

1. **Reference their specific details**: If they said "AI for hospitals", say "AI for hospitals" back to them, not "your company"
2. **Acknowledge their unique angle**: What makes THEIR story different? Highlight it.
3. **Show genuine interest**: Use phrases like "That's fascinating", "I love that", "That's a bold move"
4. **Build on their narrative**: Connect their past to their present to their future
5. **Ask story-driven follow-ups**: "What led you to that decision?" not "What's your goal?"
{covered_topics_text}{missing_required_text}
## 🚫 OFF-TOPIC DETECTION (CRITICAL)

If the user asks general knowledge questions, trivia, or anything unrelated to their professional profile:
- Set "is_off_topic": true
- Do NOT answer their question
- Provide a gentle redirect in follow_up_question

OFF-TOPIC EXAMPLES (set is_off_topic: true):
- "What's the capital of France?"
- "Can you help me write an email?"
- "Tell me a joke"
- "What's 2+2?"
- "Who invented the telephone?"
- "What's the weather like?"
- "Can you explain blockchain?"

REDIRECT TEMPLATE: "I'd love to explore that another time! Right now, let's focus on finding you great connections. [relevant profile question]"

ON-TOPIC (profile-related, set is_off_topic: false):
- "I'm a founder building an AI startup"
- "I invest in early-stage companies"
- "I'm looking for a technical co-founder"

## 🧠 INDIRECT ELICITATION TECHNIQUES

### Technique 1: Open-Ended Exploration
Ask questions that invite storytelling, which reveals multiple slots naturally.
- "Walk me through your journey to where you are now"
- "What does a typical week look like for you?"
- "Tell me about the problem you're solving"

### Technique 2: Consequential Questions
Ask about outcomes/implications to reveal underlying details.
- "What would success look like for you in the next year?"
- "If you found the perfect connection tomorrow, what would that unlock?"
- "What's holding you back from your next milestone?"

### Technique 3: Hypothetical Scenarios
Use "imagine" or "if" to get honest, detailed responses.
- "If you had unlimited resources, what would you build first?"
- "Imagine your ideal partner - what skills do they bring?"

### Technique 4: Comparative Questions
Ask them to compare/contrast to reveal preferences.
- "What's different about your approach compared to others in your space?"
- "Of all the challenges you face, which one keeps you up at night?"

### Technique 5: Reflective Probing
Dig deeper into something they mentioned to extract more slots.
- "You mentioned [X] - what led you to that decision?"
- "That's an interesting choice - how has that shaped your approach?"

## 📊 DEEP CONTEXT ANALYSIS

When analyzing user responses, look for:

1. **Explicit statements**: Direct mentions ("I'm raising $2M", "I'm a CTO")
2. **Implicit signals**: What they DON'T say, their word choices, tone
3. **Contextual inference**: "Looking for investors" = FOUNDER (not investor)
4. **Multi-slot extraction**: One sentence may fill 3-4 slots

EXAMPLE ANALYSIS:
User: "I left my corporate job last year to build an AI tool for hospitals. We have 3 pilot customers and are looking to raise our seed round."

Extract ALL of these:
- user_type: "Founder/Entrepreneur" (building a company)
- industry_focus: ["Healthcare/Biotech", "AI/ML"] (AI for hospitals)
- company_stage: "MVP" (has pilot customers)
- stage_preference: "Seed" (raising seed round)
- primary_goal: "Raise Funding" (looking to raise)
- experience_years: infer "1+ years" (left job last year)
- team_size: infer "1-3" (small team implied)

## Critical Rules
1. ALWAYS reference specific details from user's message to show you were listening (DO mirror their words with enthusiasm)
2. NEVER use generic acknowledgments like "Thanks for sharing!" or "Got it" — always personalize
3. NEVER ask for information already collected (see ALREADY COLLECTED section)
4. NO word limit - ask thoughtful questions that yield rich responses
5. If user says "looking for investors" or "raising funding" - they are a FOUNDER, not an investor
6. CEO, Founder, Co-founder = Founder/Entrepreneur
7. PRIORITIZE questions that can fill multiple missing slots at once
8. Make them feel like the most interesting person you've talked to today

## Slots to Extract
{chr(10).join(slot_descriptions)}
{already_filled_text}
## Response Format
Return valid JSON:
{{
    "is_off_topic": false,
    "extracted_slots": {{
        "slot_name": {{
            "value": "extracted value",
            "confidence": 0.0-1.0,
            "reasoning": "why you extracted this - include implicit inferences"
        }}
    }},
    "user_type_inference": "founder|investor|advisor|executive|service_provider|unknown",
    "understanding_summary": "INTERNAL ONLY - your analysis notes including implicit signals detected",
    "missing_important_slots": ["REQUIRED slots first: primary_goal, requirements, offerings, user_type, industry_focus, stage_preference, geography"],
    "follow_up_question": "Open-ended question designed to fill multiple missing slots"
}}

## follow_up_question Guidelines

Your follow-up MUST:
1. **Start with warm acknowledgment** that references SPECIFIC details they shared (not generic "Thanks for sharing!")
2. **Show genuine curiosity** with phrases like "That's fascinating", "I love that approach", "What a journey"
3. **Connect to their unique story** — don't ask generic questions, ask questions that flow from THEIR narrative
4. **Be open-ended** to invite rich responses
5. **Never be form-like** — no "What is your X?" or "Tell me about your Y"

**PERSONALIZATION TEMPLATE:**
"[Warm acknowledgment referencing their specific detail] — [genuine curiosity phrase]. [Question that flows naturally from their story]"

**EXAMPLE:**
User: "I left my corporate job to build an AI tool for hospitals"
GOOD: "That's a bold leap — leaving corporate to tackle healthcare AI. Those hospital sales cycles can be brutal. What made you confident enough to make that jump?"
BAD: "Thanks for sharing! What are your goals?"

## 🚫 FORBIDDEN SEMANTIC PATTERNS (CRITICAL)

These question topics are ALL THE SAME - if you asked ANY one, do NOT ask another:
- goals / objectives / priorities / aspirations / ambitions / vision
- achievements / accomplishments / success / hopes / dreams
- challenges / obstacles / blockers / struggles / difficulties
- skills / expertise / experience / strengths / capabilities / superpowers
- needs / requirements / support / help / gaps / looking for

If user already answered about "goals", do NOT ask about "priorities" or "aspirations" - they're the same thing!

## 🔄 "I ALREADY ANSWERED" HANDLING (CRITICAL)

If user says ANY of these:
- "I already answered"
- "I told you"
- "As I mentioned"
- "I said before"
- "Same as above"
- [Repeats previous answer]

Then you MUST:
1. Set extracted_slots to {{}} (nothing new to extract from repeat)
2. Ask about a COMPLETELY DIFFERENT topic (switch from goals→geography, or skills→stage preference)
3. Acknowledge briefly if natural ("Got it!" then new topic)

NEVER ask a variation of the same question after user signals repetition.

MUST NOT contain:
- "The user is..." or "They are..." (third person)
- Generic filler: "Nice to meet you!", "Thanks for sharing!", "Great!", "Got it."
- Form-like questions: "What is your X?", "Tell me about your Y"
- Questions semantically similar to previously asked ones

MUST contain:
- Specific reference to something unique from their message (names, companies, decisions, experiences)
- Genuine curiosity phrase ("fascinating", "love that", "bold move", "interesting")
- Natural conversational flow (not interrogation)

## Strategic Question Examples (Engaging Style)

**If user mentioned leaving a job to start something:**
"That's a brave move — walking away from stability to chase something you believe in. What was the moment you knew you had to make that leap?"

**If user mentioned a specific industry (e.g., healthtech):**
"Healthcare is such a fascinating space — complex but so much potential for impact. What drew you to that world specifically?"

**If user mentioned raising funding:**
"Fundraising can be such a rollercoaster. What's been the most surprising thing about that process so far?"

**If user mentioned a co-founder search:**
"Finding the right co-founder is like dating but with higher stakes! What qualities matter most to you in that partnership?"

**If user is an investor:**
"I love hearing what makes investors tick. What's your thesis — what gets you genuinely excited to write a check?"

**Generic engaging openers (personalize based on context):**
- "What's the big vision you're working toward? I'm curious what success looks like to you."
- "Every journey has its turning points — what was yours?"
- "What's been the biggest unlock for you so far, and what would accelerate things most?"

## Example Extraction (Engaging Style)

User already provided: name=Sarah, company=TechVenture
User says: "We're a B2B SaaS in the HR space. Been at it for 2 years with a small team. Looking to raise our Series A to expand into Europe."

Response:
{{
    "is_off_topic": false,
    "extracted_slots": {{
        "industry_focus": {{"value": ["Technology/SaaS", "Enterprise"], "confidence": 0.95, "reasoning": "B2B SaaS = Technology/SaaS + Enterprise"}},
        "experience_years": {{"value": "2 years", "confidence": 0.95, "reasoning": "Explicitly stated 'been at it for 2 years'"}},
        "team_size": {{"value": "small team", "confidence": 0.8, "reasoning": "Mentioned 'small team' - likely 2-10 people"}},
        "stage_preference": {{"value": "Series A", "confidence": 0.95, "reasoning": "Raising Series A"}},
        "geography": {{"value": ["Europe"], "confidence": 0.9, "reasoning": "Expanding into Europe - likely current market elsewhere"}},
        "primary_goal": {{"value": "Raise Funding", "confidence": 0.95, "reasoning": "Looking to raise Series A"}}
    }},
    "user_type_inference": "founder",
    "understanding_summary": "B2B SaaS founder in HR tech, 2 years in, small team, raising Series A for European expansion. Likely US/UK based currently.",
    "missing_important_slots": ["requirements", "offerings"],
    "follow_up_question": "HR tech is such a competitive space — building for 2 years and now going after Europe is impressive. What's been your secret weapon for standing out, and what kind of support would really accelerate that expansion?"
}}

## 🎯 GOOD vs BAD Follow-up Examples

**User said: "I'm a founder building an AI tool for hospitals"**

❌ BAD: "Thanks for sharing! What are your goals?"
❌ BAD: "I see. What stage is your company?"
❌ BAD: "Got it. Tell me about your requirements."

✅ GOOD: "Building AI for hospitals — that's such a fascinating but complex space to crack. Healthcare moves slowly but the potential impact is massive. What sparked your decision to take on this challenge?"

**User said: "I pivoted from sales into tech last year"**

❌ BAD: "Great! What do you do now?"
❌ BAD: "Thanks for sharing. What industry are you in?"

✅ GOOD: "That's a bold pivot — sales to tech is quite the journey. I imagine your sales background gives you a unique edge on the product side. How has that experience shaped what you're building?"

Extract ALL inferable information. Ask questions that reveal what's STILL MISSING."""

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
            understanding_summary=response_data.get("understanding_summary", ""),
            is_off_topic=response_data.get("is_off_topic", False)
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

    def _user_wants_to_finish(
        self,
        message: str,
        already_filled: Dict[str, Any]
    ) -> bool:
        """
        Detect if user wants to end onboarding.

        BUG-001 FIX: This prevents the LLM from generating follow-up questions
        when the user explicitly signals they're done.

        Args:
            message: User's message
            already_filled: Already collected slots (need minimum for valid profile)

        Returns:
            True if user wants to finish and has enough data
        """
        # Completion phrases that indicate user wants to finish
        completion_phrases = [
            "done", "that's all", "that's everything", "i'm done", "im done",
            "show me my matches", "find my matches", "i'm ready", "im ready",
            "start matching", "no more", "nothing else", "finish", "complete",
            "let's start", "lets start", "proceed", "move on", "that covers",
            "ready to match", "see my matches"
        ]

        msg_lower = message.lower().strip()

        # Check if message contains completion signal
        has_completion_signal = any(phrase in msg_lower for phrase in completion_phrases)

        if not has_completion_signal:
            return False

        # Only allow completion if we have minimum viable profile (at least 3 slots)
        filled_count = len([v for v in already_filled.values() if v])
        if filled_count < 3:
            logger.info(f"Completion signal detected but only {filled_count} slots filled (need 3)")
            return False

        return True

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

        # Fallback response generation (engaging style)
        if extraction_result.missing_slots:
            missing = extraction_result.missing_slots[:2]
            return f"I'm loving hearing your story! I'm curious about your {' and '.join(missing)} — what would you say defines your approach there?"

        return "I feel like I'm getting a great picture of who you are. Ready to see who we can connect you with?"


# Global instance
llm_slot_extractor = LLMSlotExtractor()
