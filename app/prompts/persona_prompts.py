"""
Prompt templates for persona generation.
"""
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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

2. requirements: 3–4 sentences focusing on what this individual ACTUALLY seeks based on their role

3. offerings: 3–4 sentences focusing on what this individual can ACTUALLY provide

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
"""


def build_persona_chain(llm):
    """Build the persona generation chain."""
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template=PERSONA_TEMPLATE,
        input_variables=["combined_data", "json_schema"],
    )
    # Chain: Prompt -> LLM -> JSON parser
    chain = prompt | llm | parser
    return chain


def combine_user_data(questions: list, resume_text: str) -> str:
    """Combine user questions and resume text into a single string."""
    combined = []
    
    # Add questions section
    if questions:
        combined.append("User Questions:")
        for i, q in enumerate(questions, 1):
            if isinstance(q, dict):
                prompt = q.get('prompt', '')
                answer = q.get('answer', '')
                combined.append(f"{i}. {prompt}\n   Answer: {answer}")
            else:
                combined.append(f"{i}. {q}")
        combined.append("")
    
    # Add resume section
    if resume_text:
        combined.append("Resume Content:")
        combined.append(resume_text)
    
    return "\n".join(combined)
