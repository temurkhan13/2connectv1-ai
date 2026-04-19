"""
Prompt templates for persona generation.
"""
import re
import json
import logging
from typing import Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

logger = logging.getLogger(__name__)

# JSON schema for persona output with requirements and offerings
# NOTE: "strategy" replaces "investment_philosophy" to be role-agnostic
# For investors: strategy = investment approach
# For founders: strategy = business/growth approach
# For advisors: strategy = advisory approach
PERSONA_JSON_SPEC = {
    "type": "object",
    "properties": {
        "persona": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "archetype": {"type": "string"},
                "designation": {"type": "string"},
                "experience": {"type": "string"},
                "focus": {"type": "string"},
                "profile_essence": {"type": "string"},
                "strategy": {"type": "string"},
                "what_theyre_looking_for": {"type": "string"},
                "engagement_style": {"type": "string"}
            },
            "required": ["name", "archetype","designation", "experience", "focus", "profile_essence", "strategy", "what_theyre_looking_for", "engagement_style"]
        },
        "requirements": {"type": "string"},
        "offerings": {"type": "string"}
    },
    "required": ["persona", "requirements", "offerings"]
}

# Prompt template for persona generation with requirements and offerings
# PERSONA_TEMPLATE = """
# You are an expert persona generator. Based on the combined data below (questions + resume text), generate THREE outputs: a professional persona, their requirements, and their offerings.

# IMPORTANT: If the provided questions and resume are blank, irrelevant, or do not contain enough meaningful information, do NOT generate a persona. Instead, return an error or a message stating persona generation is not possible due to insufficient data.

# Output VALID JSON ONLY matching this schema: {json_schema}

# Generate the following:

# 1. **persona**: An investor persona with these fields:
#    - name: A creative, memorable persona name (e.g., "The Calculated Catalyst", "The Strategic Visionary")
#    - archetype: A descriptive archetype (e.g., "Seasonal Tech-Driven Investor", "Growth-Focused Strategist")
#    - designation: The user's professional designation or job title (e.g., "Software Engineer", "Data Scientist"). Extract ONLY from the provided data. If not explicitly mentioned, use "Not specified". Do NOT infer or make up designations.
#    - experience: Description of experience based ONLY on what is provided. Do NOT add years of experience unless explicitly stated in the data. If not mentioned, use "Not specified".
#    - focus: Key focus areas separated by " | " (e.g., "Early-Stage Startups | MVP-Ready Products | Tech-Centric Visionaries")
#    - profile_essence: 2-3 sentences describing the persona's core characteristics and approach
#    - investment_philosophy: 3-4 bullet points describing their investment approach and values
#    - what_theyre_looking_for: Specific sectors, stages, and criteria they seek
#    - engagement_style: How they prefer to interact and engage with opportunities

# 2. **requirements**: What this person is looking for, needs, or seeking (2-3 sentences):
#    - Extract from their stated goals, objectives, and what they want to achieve
#    - Based on questions about funding needs, partnership goals, learning objectives, etc.
#    - Focus on what they need from others

# 3. **offerings**: What this person can provide, offer, or give to others (2-3 sentences):
#    - Extract from their professional background, skills, experience, and expertise
#    - Based on resume content and their stated strengths
#    - Focus on what they can offer to others

# Guidelines:
# - Be specific and grounded ONLY in the provided data
# - Do NOT make up, infer, or hallucinate information not present in the input
# - Requirements and offerings should be distinct and complementary
# - Use concrete, actionable language
# - Write in third person
# - Make all outputs sound professional and realistic
# - For designation and experience: Use ONLY what is explicitly stated. Do not infer years of experience or job titles.

# Combined Data:
# {combined_data}
# """
PERSONA_TEMPLATE = """
You are an expert persona generation system. Using the combined data provided below (user questions + resume text), generate THREE structured outputs: a professional persona, their requirements, and their offerings.

Output STRICTLY in VALID JSON format matching this schema: {json_schema}

CRITICAL - ROLE DETECTION: First identify the user's role from the data:
- If they mention "raising funding", "seeking investment", "looking for investors", "CEO of a startup", "building a company" → They are a FOUNDER/ENTREPRENEUR seeking funding
- If they mention "investing", "portfolio", "check size", "deal flow", "angel investor", "VC" → They are an INVESTOR
- If they mention "advisory", "consulting", "mentoring", "board member" → They are an ADVISOR
- BUG-038 FIX: If they mention "recruiting", "recruiter", "talent acquisition", "headhunter", "placing candidates", "hiring for clients", "staffing", "talent partner" → They are a RECRUITER/TALENT PROFESSIONAL (NOT entrepreneur even if they say "founder of recruiting firm")
- BUG-038 FIX: If they mention "agency", "consulting firm", "service provider", "client work", "serving companies" → They are a SERVICE PROVIDER (NOT entrepreneur)
- IMPORTANT: "founder of X" where X is a service business (recruiting, consulting, agency) = SERVICE PROVIDER, not startup founder
- If they mention "job search", "looking for a job", "seeking employment", "career change", "open to opportunities", "target role", "salary expectations", "notice period" → They are a JOB SEEKER/CANDIDATE
- If they mention "partnership", "strategic alliance", "joint venture", "distribution partner", "co-marketing" → They are seeking BUSINESS PARTNERSHIP
- Otherwise, infer from context or default to professional seeking connections

ROLE-AWARE OUTPUT - The "strategy" field MUST match their role:
- For FOUNDERS: Business strategy, growth plans, market approach (NOT investment philosophy)
- For INVESTORS: Investment thesis, check size preferences, sector focus
- For ADVISORS: Advisory approach, areas of expertise, engagement style
- For JOB SEEKERS: Career goals, target roles, growth aspirations, work preferences
- For PARTNERSHIP SEEKERS: Partnership objectives, collaboration style, mutual value creation

Mandatory fallback behavior:
- If the provided information is blank, irrelevant, or insufficient, you MUST still output VALID JSON.
- For any missing fields, output the string "Not specified".
- Do not infer or fabricate details beyond the given input.

IMPORTANT — USE BOTH SOURCES FOR ALL FIELDS:
When a resume/CV is provided, use it to ENRICH every persona field — not just offerings.
- designation: Extract EXACT job title from resume (e.g., "VP of Engineering at Stripe") instead of generic titles
- experience: Extract EXACT years and companies from resume work history
- focus: Combine resume expertise areas with Q&A stated interests
- profile_essence: Weave resume achievements into the narrative
- strategy: Inform from both stated goals (Q&A) and demonstrated track record (resume)
- The resume makes the persona SPECIFIC and CREDIBLE instead of generic

Generate the following outputs — WRITE EXTENSIVELY, aim for 1,500-2,000 total words across all fields:

1. persona:
   - name: Concise, creative title reflecting their ACTUAL role (e.g., "The Growth-Focused Founder" for founders, NOT "The Strategic Investor")
   - archetype: Descriptive classification matching their role
   - designation: Explicit job title from resume or input; if missing, "Not specified". Prefer resume title over Q&A.
   - experience: EXACT years/description from resume work history or input. If resume shows 3 jobs spanning 2018-2026, write "8 years". If not explicitly stated anywhere, write "Not specified". NEVER inflate.
   - focus: Key areas separated by " | " — combine resume skills/domains with Q&A stated interests. Be specific and detailed (e.g., "Cross-border B2B Payments | ML-powered Risk Scoring | Emerging Market Compliance" not just "FinTech").
   - profile_essence: 8–12 sentences. This is the MOST IMPORTANT field — it becomes the embedding source. Write a rich, detailed narrative covering: who they are, their journey, what they've built/achieved, their expertise areas, their working style, what drives them, and what makes them unique. Reference specific companies, roles, numbers, markets, technologies, and achievements. Use the full conversation text for narrative detail — don't just summarize slot values.
   - strategy: 6–8 detailed bullet points describing their approach with specifics. Use resume track record and conversation detail to inform strategy. Each bullet should be 1-2 sentences with concrete details.
   - what_theyre_looking_for: 4–6 sentences detailing specifically what they seek. Include the type of person, their ideal characteristics, the kind of value exchange they want, and any stated preferences about geography, stage, industry, or expertise.
   - engagement_style: 2-3 sentences about how they prefer to interact — communication style, meeting cadence, relationship depth, formality level.

2. requirements: 6–8 sentences focusing on what this individual ACTIVELY SEEKS from connections
   - Extract primarily from conversation text and Q&A stated GOALS, NEEDS, and what they're LOOKING FOR
   - These are GAPS they want to FILL — things they DON'T have
   - Be SPECIFIC about what type of person/resource they need, in what industry, at what stage, in what geography
   - Include context about WHY they need this (e.g., "needs Series A investors because seed round closes in Q2 and they need to maintain momentum")
   - Resume context helps qualify requirements
   - RULE: If it describes what they CAN DO or HAVE DONE, it's offerings, NOT requirements

3. offerings: 6–8 sentences focusing on what this individual can PROVIDE to connections
   - Extract from RESUME primarily: work history, skills, achievements, companies, education, network
   - Also include capabilities mentioned in conversation and Q&A answers
   - Be EXTREMELY SPECIFIC: "10 years scaling B2B SaaS payment infrastructure at Stripe and Adyen, managing teams of 40+ engineers across 3 time zones" beats "experience in SaaS"
   - Include: industry connections (name specific networks), domain expertise (specific sub-domains), technical skills (specific technologies), leadership experience (team sizes, outcomes), specific achievements (metrics, milestones)
   - Describe the VALUE someone gets from connecting with this person
   - RULE: If it describes what they WANT or NEED, it's requirements, NOT offerings

Generation rules:
- Use ONLY the provided input; never infer beyond it.
- NEVER describe a founder as an investor or vice versa.
- If someone says "I'm looking for investors", they are a FOUNDER, not an investor.
- Maintain professional, realistic tone.
- Keep requirements and offerings distinct.
- Always follow the JSON schema exactly.
- For any missing details, use "Not specified".
- ANONYMIZATION-FRIENDLY PROSE: In profile_essence, requirements, offerings, strategy,
  what_theyre_looking_for, and engagement_style — reference the user's name at most
  ONCE per section (preferably not at all if the narrative reads naturally without it).
  After any name mention, use pronouns (they/their/them) or role-based references
  ("this founder", "this recruiter", "this investor") for subsequent references.
  This keeps prose natural while minimizing name repetition — important for the
  Discover view which anonymizes other users' profiles by stripping names.
- PUBLIC PRIVACY RULE (Apr-19 Issue 8 per [[Apr-18]] Follow-up 27):
  In profile_essence, requirements, offerings, strategy, and what_theyre_looking_for
  — DO NOT name specific real-world companies or specific real-world people the
  user has worked with. Use categorical descriptors that preserve the signal:
    * "at Stripe" → "at a leading payments infrastructure company"
    * "at Google" → "at a major tech company"
    * "ex-Monzo" → "from a prominent UK digital bank"
    * "scaled Ramp from Series A to C" → "scaled a Series A fintech to Series C"
    * "co-founded Oscar Health" → "co-founded a health-insurance scale-up"
    * "mentored by Andy Rachleff" → "mentored by a well-known venture partner"
  Keep ALL OTHER SPECIFICS intact: years of experience, skill areas, industry focus,
  fundraising amounts, deal counts, cheque sizes, team sizes, geographies, stage
  preferences, sector thesis, metrics. The categorical descriptor retains matching
  signal while making the text safe for the Discover page where identity must not
  leak.

  ** CONSISTENCY MANDATE (Apr-19 Brian Limba test): APPLY THIS RULE TO EVERY
  MENTION OF EVERY REAL-WORLD NAMED ENTITY IN EVERY ONE OF THE FIVE FIELDS
  ABOVE — FROM THE FIRST WORD TO THE LAST. **
  Once you've chosen a categorical descriptor for a company or person (e.g.
  "a leading payments infrastructure company" for Stripe), USE THAT SAME
  DESCRIPTOR every subsequent time you refer to that entity in the same field
  or across fields — NEVER slip back to the literal name. Do not let the rule
  relax as the text grows longer. The LLM's tendency to revert to the verbatim
  name on the N-th mention after using a categorical descriptor early on IS
  the failure mode we are explicitly preventing. Paragraphs 1, 2, AND the
  closing sentences of offerings get the same treatment.

  SELF-CHECK before finalizing your JSON response:
  Re-scan the five public fields (profile_essence, requirements, offerings,
  strategy, what_theyre_looking_for) for any real-world company name, fund
  name, accelerator name, institution name, or person name that appears
  verbatim. If you find any — replace with the same categorical descriptor
  you used earlier in the text. "Sequoia seed network" → use the categorical
  equivalent ("premier early-stage venture fund's network"). "Priya" (a
  co-founder's first name) → "the co-founder". This check is non-negotiable;
  a single literal name slipping through defeats the whole privacy guarantee.

  EXCEPTION — the `persona_archetype` and `persona_designation` fields ARE allowed
  to contain specific companies because they are used for SCORING only and not
  displayed cross-user. Use the persona title as the place to be fully specific;
  use the essence/requirements/offerings as the place to be categorically specific.
- CRITICAL: The FULL ONBOARDING CONVERSATION section contains the user's detailed responses in their own words. Use this as your PRIMARY source for narrative detail. The Q&A section has compressed slot values — use those only as factual anchors.
- The profile_essence, requirements, and offerings fields are used to generate embedding vectors for matching. LONGER AND MORE DETAILED = BETTER MATCHES. Do not compress or summarize — expand and elaborate using all available data.

Combined Input Data:
{combined_data}

CRITICAL OUTPUT INSTRUCTION:
Your response MUST be ONLY valid JSON. Do NOT include any explanatory text, preamble, or commentary.
Do NOT start with phrases like "Based on the provided..." or "Here are the outputs..."
Start your response DIRECTLY with the opening curly brace {{ and end with the closing curly brace }}.
"""


class RobustJsonOutputParser(JsonOutputParser):
    """
    Enhanced JSON parser that handles LLM responses wrapped in markdown text.

    BUG-015 FIX: Claude sometimes wraps JSON in explanatory text like:
    "Based on the provided input data, here are the outputs:
    {
      "persona": {...}
    }"

    This parser extracts JSON from such responses before parsing.
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown-wrapped JSON.

        Args:
            text: LLM response that may contain JSON wrapped in markdown

        Returns:
            Parsed JSON dict

        Raises:
            OutputParserException: If no valid JSON found
        """
        # Try direct parse first (fast path for well-behaved responses)
        try:
            return super().parse(text)
        except Exception as direct_error:
            logger.debug(f"Direct JSON parse failed, trying extraction: {direct_error}")

        # BUG-015 FIX: Extract JSON from markdown text
        # Match the outermost {...} structure (handles nested objects)
        json_match = re.search(r'\{[\s\S]*\}', text)

        if json_match:
            try:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                logger.info("BUG-015 FIX: Successfully extracted JSON from markdown-wrapped response")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"BUG-015 FIX: Found JSON-like text but parse failed: {e}")
                # Fall through to additional attempts

        # BUG-027 FIX: Try to extract JSON from code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if code_block_match:
            try:
                json_str = code_block_match.group(1)
                parsed = json.loads(json_str)
                logger.info("BUG-027 FIX: Successfully extracted JSON from code block")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"BUG-027 FIX: Code block JSON parse failed: {e}")

        # BUG-027 ENHANCED FIX: Always generate fallback when all JSON parsing attempts fail
        # This handles:
        # 1. No JSON at all (descriptive text only)
        # 2. Malformed/truncated JSON
        # 3. Partial JSON with missing closing braces
        # NOTE: _generate_fallback_persona() NEVER fails - it always returns a valid dict
        logger.warning(f"BUG-027 FIX: All JSON parsing attempts failed, generating fallback persona. First 200 chars: {text[:200]}")
        return self._generate_fallback_persona(text)

    def _generate_fallback_persona(self, text: str) -> Dict[str, Any]:
        """
        BUG-027 ENHANCED FIX: Generate a minimal fallback persona when LLM fails to output JSON.

        This extracts what information we can from the descriptive text and creates
        a valid JSON structure to prevent complete failure.

        NOTE: This method MUST NOT fail - it always returns a valid persona dict.
        """
        # Try to extract any useful information from the text
        profession = "Not specified"
        try:
            # Look for patterns like "professional seeking employment" or "public accounting"
            profession_match = re.search(r'(?:professional|expert|specialist|seeking|working in|field of)\s+([^.,]+)', text or "", re.IGNORECASE)
            if profession_match:
                profession = profession_match.group(1).strip()[:50]
        except Exception:
            pass  # Use default "Not specified"

        # This structure ALWAYS succeeds - no exceptions possible
        fallback = {
            "persona": {
                "name": "Profile Under Review",
                "archetype": "Professional",
                "designation": profession,
                "experience": "Not specified",
                "focus": "Not specified",
                "profile_essence": "Profile information is being processed. Please check back shortly.",
                "strategy": "Not specified",
                "what_theyre_looking_for": "Not specified",
                "engagement_style": "Not specified"
            },
            "requirements": "Profile requirements are being processed.",
            "offerings": "Profile offerings are being processed."
        }
        logger.info("BUG-027 FIX: Generated fallback persona (LLM output was not valid JSON)")
        return fallback


def build_persona_chain(llm):
    """Build the persona generation chain with robust JSON parsing."""
    # BUG-015 FIX: Use RobustJsonOutputParser instead of standard JsonOutputParser
    # This handles LLM responses that wrap JSON in explanatory markdown text
    parser = RobustJsonOutputParser()
    prompt = PromptTemplate(
        template=PERSONA_TEMPLATE,
        input_variables=["combined_data", "json_schema"],
    )
    # Chain: Prompt -> LLM -> Robust JSON parser
    chain = prompt | llm | parser
    return chain


def combine_user_data(questions: list, resume_text: str, conversation_text: str = "") -> str:
    """
    Combine user data from multiple sources into a single string for persona generation.

    Data sources (in priority order for richness):
    - Full conversation: Raw onboarding dialogue — richest narrative source
    - Q&A: Extracted slot question/answer pairs — structured but compressed
    - Resume: Background, experience, skills, achievements

    The full conversation text preserves the user's natural language and detail
    that gets lost when compressed into short slot values. This produces
    significantly richer AI summaries and better embeddings.
    """
    combined = []

    # PRIORITY 1: Full conversation text (richest source — user's own words)
    if conversation_text and conversation_text.strip():
        combined.append("=" * 60)
        combined.append("FULL ONBOARDING CONVERSATION")
        combined.append("The complete dialogue — contains the user's detailed, natural responses.")
        combined.append("This is the PRIMARY source for understanding who they are and what they need.")
        combined.append("=" * 60)
        combined.append(conversation_text.strip())
        combined.append("")

    # PRIORITY 2: Structured Q&A (slot extractions — supplement if conversation missing)
    if questions:
        combined.append("=" * 60)
        combined.append("EXTRACTED Q&A RESPONSES (structured data)")
        combined.append("Condensed slot values extracted from the conversation above.")
        combined.append("Use these as factual anchors, but prefer the conversation for narrative detail.")
        combined.append("=" * 60)
        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                prompt = q.get('prompt', '')
                answer = q.get('answer', '')
                combined.append(f"{i}. {prompt}\n   Answer: {answer}")
            else:
                combined.append(f"{i}. {q}")
        combined.append("")

    # PRIORITY 3: Resume / CV
    if resume_text:
        combined.append("=" * 60)
        combined.append("RESUME / CV / PROFESSIONAL BACKGROUND")
        combined.append("Their work history, skills, achievements, education, and expertise.")
        combined.append("Use this to enrich offerings, designation, experience, and strategy.")
        combined.append("=" * 60)
        combined.append(resume_text)

    return "\n".join(combined)
