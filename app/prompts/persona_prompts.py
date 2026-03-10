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
- If they mention "raising funding", "seeking investment", "looking for investors", "founder", "CEO of a startup", "building a company" → They are a FOUNDER/ENTREPRENEUR seeking funding
- If they mention "investing", "portfolio", "check size", "deal flow", "angel investor", "VC" → They are an INVESTOR
- If they mention "advisory", "consulting", "mentoring", "board member" → They are an ADVISOR
- Otherwise, infer from context or default to professional seeking connections

ROLE-AWARE OUTPUT - The "strategy" field MUST match their role:
- For FOUNDERS: Business strategy, growth plans, market approach (NOT investment philosophy)
- For INVESTORS: Investment thesis, check size preferences, sector focus
- For ADVISORS: Advisory approach, areas of expertise, engagement style

Mandatory fallback behavior:
- If the provided information is blank, irrelevant, or insufficient, you MUST still output VALID JSON.
- For any missing fields, output the string "Not specified".
- Do not infer or fabricate details beyond the given input.

Generate the following outputs:

1. persona:
   - name: Concise, creative title reflecting their ACTUAL role (e.g., "The Growth-Focused Founder" for founders, NOT "The Strategic Investor")
   - archetype: Descriptive classification matching their role
   - designation: Explicit job title from input; if missing, "Not specified"
   - experience: Description from input; if missing, "Not specified"
   - focus: Key areas separated by " | "
   - profile_essence: 3–4 sentences, grounded in input, accurately reflecting their role
   - strategy: 3–4 bullet points describing their approach (business strategy for founders, investment thesis for investors)
   - what_theyre_looking_for: What they ACTUALLY seek (investors for founders, deals for investors)
   - engagement_style: Preferred communication or collaboration approach

2. requirements: 3–4 sentences focusing on what this individual ACTIVELY SEEKS from connections
   CRITICAL DISTINCTION:
   - Extract ONLY from their stated GOALS, NEEDS, and what they're LOOKING FOR in answers
   - These are GAPS they want to FILL - things they DON'T have
   - Examples: "looking for investors", "need help with marketing", "seeking advisors", "want introductions to X"
   - NEVER include capabilities from their resume or background here - that's offerings!
   - RULE: If it describes what they CAN DO or HAVE DONE, it's offerings, NOT requirements

3. offerings: 3–4 sentences focusing on what this individual can PROVIDE to connections
   CRITICAL DISTINCTION:
   - Extract from their BACKGROUND, EXPERIENCE, SKILLS, ACHIEVEMENTS, and NETWORK
   - These come from their resume/CV and professional history
   - Examples: "20 years in healthcare", "connections to VCs", "built companies that raised $XM", "expertise in X"
   - These are capabilities they ALREADY HAVE - value they bring to others
   - RULE: If it describes what they WANT or NEED, it's requirements, NOT offerings

Generation rules:
- Use ONLY the provided input; never infer beyond it.
- NEVER describe a founder as an investor or vice versa.
- If someone says "I'm looking for investors", they are a FOUNDER, not an investor.
- Maintain professional, realistic tone.
- Keep requirements and offerings distinct.
- Always follow the JSON schema exactly.
- For any missing details, use "Not specified".

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


def combine_user_data(questions: list, resume_text: str) -> str:
    """
    Combine user questions and resume text into a single string.

    IMPORTANT: Sections are clearly marked to help LLM distinguish:
    - Q&A Section: Contains user's stated goals/needs → PRIMARY SOURCE for REQUIREMENTS
    - Resume Section: Contains background/experience → PRIMARY SOURCE for OFFERINGS
    """
    combined = []

    # Add questions section with clear labeling
    if questions:
        combined.append("=" * 60)
        combined.append("USER Q&A RESPONSES (SOURCE: Extract REQUIREMENTS from this section)")
        combined.append("What they said they're looking for, need, want to achieve")
        combined.append("=" * 60)
        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                prompt = q.get('prompt', '')
                answer = q.get('answer', '')
                combined.append(f"{i}. {prompt}\n   Answer: {answer}")
            else:
                combined.append(f"{i}. {q}")
        combined.append("")

    # Add resume section with clear labeling
    if resume_text:
        combined.append("=" * 60)
        combined.append("RESUME/BACKGROUND (SOURCE: Extract OFFERINGS from this section)")
        combined.append("What they have done, can do, their expertise and network")
        combined.append("=" * 60)
        combined.append(resume_text)

    return "\n".join(combined)
