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
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from anthropic import Anthropic
from app.services.use_case_templates import get_onboarding_slots

logger = logging.getLogger(__name__)


# =============================================================================
# MULTI-VECTOR DIMENSIONS FOR STEERING
# =============================================================================
# These dimensions are critical for match quality. The LLM should prioritize
# collecting these before diving deep into detail questions.

MULTI_VECTOR_DIMENSION_SLOTS = {
    "primary_goal": "What's your main objective on the platform?",
    "industry_focus": "What industries or sectors do you focus on?",
    "stage_preference": "What company stage are you most interested in?",
    "geography": "What regions or markets are you focused on?",
    "engagement_style": "What kind of relationship would be most valuable for you?",
    "dealbreakers": "Anything that would be a clear 'not for me'?"
}

# Maximum questions to ask on a single topic before forcing switch
MAX_QUESTIONS_PER_TOPIC = 2

# =============================================================================
# BUG-045 FIX: OBJECTIVE-SPECIFIC SLOT FILTERING
# =============================================================================
# These slots should ONLY be extracted when the user's primary_goal matches.
# Prevents confusing extractions like "Sales" for a CTO seeking funding.

# Slots specific to HIRING objective - exclude for non-hiring users
HIRING_SPECIFIC_SLOTS = {"role_type", "seniority_level", "remote_preference", "compensation_range"}

# Slots specific to COFOUNDER objective - exclude for non-cofounder users
COFOUNDER_SPECIFIC_SLOTS = {"skills_have", "skills_need", "commitment_level", "equity_expectations"}

# Slots specific to MENTORSHIP objective
MENTORSHIP_SPECIFIC_SLOTS = {"mentorship_areas", "mentorship_format", "mentorship_commitment"}

# BUG-074: Slots specific to JOB_SEARCH objective (candidates seeking jobs)
# Reuses HIRING slots (role_type, seniority_level, remote_preference, compensation_range)
# Plus skills_have from COFOUNDER (what skills the candidate brings)
JOB_SEARCH_SPECIFIC_SLOTS = {"role_type", "seniority_level", "remote_preference", "compensation_range", "skills_have"}

# BUG-075: Slots specific to SERVICES objective (service providers)
SERVICES_SPECIFIC_SLOTS = {"service_type", "engagement_style", "budget_range"}

# Map primary_goal values to specific slot sets
OBJECTIVE_SLOT_MAPPING = {
    "hiring": HIRING_SPECIFIC_SLOTS,
    "hire talent": HIRING_SPECIFIC_SLOTS,  # BUG-075: Added explicit option
    "find co-founder": COFOUNDER_SPECIFIC_SLOTS,
    "seek mentorship": MENTORSHIP_SPECIFIC_SLOTS,
    "find new job": JOB_SEARCH_SPECIFIC_SLOTS,  # BUG-074: Job seekers get job-related slots
    "offer services": SERVICES_SPECIFIC_SLOTS,  # BUG-075: Service providers
    "launch product": set(),  # BUG-075: Product launch uses general slots
}


def filter_slots_by_objective(slot_definitions: Dict, primary_goal: Optional[str]) -> Dict:
    """
    BUG-045 FIX: Filter slot definitions based on user's primary objective.

    Prevents extracting irrelevant slots (e.g., hiring slots for founders seeking funding).

    Args:
        slot_definitions: Full SLOT_DEFINITIONS dictionary
        primary_goal: User's primary goal (e.g., "Raise Funding", "Find Co-founder")

    Returns:
        Filtered slot definitions with objective-irrelevant slots removed
    """
    if not primary_goal:
        return slot_definitions

    # Normalize primary_goal for comparison
    goal_lower = primary_goal.lower()

    # Collect slots to EXCLUDE based on what's NOT the user's goal
    slots_to_exclude = set()

    # BUG-074 FIX: Job seekers ALSO need hiring-related slots (role_type, seniority, compensation)
    # "Find New Job" users should get these slots exposed
    is_job_seeker = "job" in goal_lower or "career" in goal_lower or "employment" in goal_lower
    # BUG-075 FIX: Include "hire" and "talent" for Hire Talent goal
    is_hiring = "hiring" in goal_lower or "recruit" in goal_lower or "hire" in goal_lower or "talent" in goal_lower
    # BUG-075: Detect service providers
    is_service_provider = "service" in goal_lower or "consult" in goal_lower or "agency" in goal_lower or "freelance" in goal_lower

    # If NOT hiring AND NOT job seeking, exclude hiring-specific slots
    if not is_hiring and not is_job_seeker:
        slots_to_exclude.update(HIRING_SPECIFIC_SLOTS)

    # If NOT seeking co-founder, exclude co-founder slots
    # BUT job seekers keep skills_have (what skills they bring)
    if "co-founder" not in goal_lower and "cofounder" not in goal_lower:
        if is_job_seeker:
            # Job seekers keep skills_have but not the rest
            slots_to_exclude.update(COFOUNDER_SPECIFIC_SLOTS - {"skills_have"})
        else:
            slots_to_exclude.update(COFOUNDER_SPECIFIC_SLOTS)

    # If NOT seeking mentorship, exclude mentorship slots
    if "mentor" not in goal_lower:
        slots_to_exclude.update(MENTORSHIP_SPECIFIC_SLOTS)

    # BUG-075: If NOT offering services, exclude service-specific slots
    if not is_service_provider:
        slots_to_exclude.update(SERVICES_SPECIFIC_SLOTS)

    # Filter out excluded slots
    filtered = {k: v for k, v in slot_definitions.items() if k not in slots_to_exclude}

    if slots_to_exclude:
        logger.debug(f"BUG-045 FIX: Excluded {len(slots_to_exclude)} objective-specific slots for goal '{primary_goal}'")

    return filtered


@dataclass
class LLMExtractedSlot:
    """A slot extracted by the LLM."""
    name: str
    value: Any
    confidence: float
    reasoning: str  # Why the LLM extracted this value


@dataclass
class LLMExtractionResult:
    """Result of LLM slot extraction.

    BUG-071 FIX: follow_up_question is now optional (default empty).
    Question generation is handled by separate LLMQuestionGenerator service.

    BUG-092 FIX: Added is_completion_signal - LLM detects if user wants to finish.
    Previously used dumb substring matching ("done" in message) which caused
    false positives like "done Africa deals" blocking extraction.
    """
    extracted_slots: Dict[str, LLMExtractedSlot]
    user_type_inference: str  # "founder", "investor", "advisor", etc.
    missing_slots: List[str]  # What's still needed
    understanding_summary: str  # Brief summary of what LLM understood
    is_off_topic: bool  # True if user asked off-topic/general knowledge question
    is_completion_signal: bool = False  # BUG-092: True if user explicitly wants to finish onboarding
    follow_up_question: str = ""  # DEPRECATED: Now handled by LLMQuestionGenerator
    acknowledged_slots: List[str] = field(default_factory=list)  # Slots mentioned by user but deferred due to BUG-088 limit


# Slot definitions for the LLM prompt
SLOT_DEFINITIONS = {
    "user_type": {
        "description": "The user's primary role in the startup ecosystem",
        "options": ["Founder/Entrepreneur", "Angel Investor", "VC Partner", "Corporate Executive", "Mentor/Advisor", "Service Provider", "Recruiter", "Job Seeker/Candidate"],
        "extraction_hint": "IMPORTANT: If user mentions 'looking for investors', 'seeking funding', 'raising a round', they are a FOUNDER seeking investors, NOT an investor themselves. Only classify as investor if they explicitly say they INVEST money in startups. If user mentions 'looking for a job', 'new role', 'career change', 'find new job', 'next opportunity', 'looking for a CTO role', 'seeking a co-founder position', 'want to join as CTO', 'looking for my next challenge', classify as JOB SEEKER/CANDIDATE — even if they mention 'co-founder' or 'CTO' as the TARGET ROLE they're seeking. Only classify as Founder/Entrepreneur if they ALREADY ARE a founder/CEO building their own company. If user runs a recruitment firm, staffing agency, or does headhunting/talent acquisition, classify as RECRUITER not Service Provider."
    },
    "primary_goal": {
        "description": "What the user wants to achieve on the platform",
        "options": ["Raise Funding", "Find Co-founder", "Seek Mentorship", "Offer Mentorship", "Explore Partnerships", "Invest in Startups", "Offer Services", "Recruit", "Find New Job", "Seek Networking", "Hire Talent", "Launch Product"],
        "extraction_hint": (
            # Apr-25 F/u 20 rewrite — semantic disambiguation tree prepended to
            # keyword mapping. Canary of 11 Seek Networking users showed 8/11
            # were drift cases — Seek Networking had become a dumping ground
            # for users whose operational intent was burrowed under relational
            # framing ("connect", "meet", "network"). [[Rules/CODING-DISCIPLINE]]
            # Rule 5 fix: reason about the concept, not substrings.

            "PRIORITY DISAMBIGUATION — apply THESE FIRST, before any keyword matching below.\n\n"

            "QUESTION 1 — Does the user have a SPECIFIC OPERATIONAL PROJECT they're actively running or actively progressing? A project is operational if there's a concrete thing happening NOW: a shipped product with users, an active fundraise, a role they're hiring for, a job search underway, an M&A mandate, a specific set of people they need to reach for a BUSINESS purpose (distribution targets, acquirers, speakers for an event, regulators, etc.).\n"
            "   • \"Launched 3 weeks back, 11 users, converting to paid\" → Launch Product\n"
            "   • \"Raising $2M seed, closing in two months\" → Raise Funding\n"
            "   • \"Looking for investors actively deploying into [sector]\" → Raise Funding\n"
            "   • \"Hiring a CTO / Head of Product\" → Hire Talent\n"
            "   • \"Job-hunting / exploring my next role\" → Find New Job\n"
            "   • \"Seeking founders building in [specific segment] for acquisition\" → Invest in Startups (M&A counts)\n"
            "   • \"Need introductions to FDA reviewers / regulators for my trial\" → Launch Product (operational need for a product in market)\n"
            "   • \"Recruiting speakers for my conference\" → Recruit\n"
            "   • \"Need to reach training and placement officers at universities\" → Explore Partnerships (B2B outreach to a named cohort)\n"
            "IF QUESTION 1 HITS A MATCH: that goal wins EVEN IF the user also describes their approach as \"connect\", \"meet peers\", \"build relationships\", \"network\" — those words describe the HOW, not the WHY. The operational project is the WHY.\n\n"

            "QUESTION 2 — Does the user explicitly describe wanting a co-founder? Accept SOFT framings, not just literal \"I need a cofounder\":\n"
            "   • \"Open to potential co-founders\" → Find Co-founder\n"
            "   • \"Would consider the right cofounder\" → Find Co-founder\n"
            "   • \"Exploring / looking for a technical partner\" → Find Co-founder\n"
            "   • \"Want someone to build with\" → Find Co-founder\n"
            "IF QUESTION 2 HITS: Find Co-founder wins over Seek Networking, even if the cofounder intent is mentioned alongside other goals.\n\n"

            "QUESTION 3 — Is the user GIVING or RECEIVING mentorship?\n"
            "   • Receiving: \"find a mentor\", \"looking for guidance\", \"need advice\", \"learn from operators who've done X\" → Seek Mentorship\n"
            "   • Giving: \"want to mentor\", \"guide others\", \"give back\", \"help founders\", \"coach\" → Offer Mentorship\n\n"

            "QUESTION 4 — ONLY if questions 1-3 all return NO: is the user's primary framing broad relationship-building for its own sake, with NO specific operational project in market? Examples of LEGITIMATE Seek Networking:\n"
            "   • VC/angel wanting peer investor network for deal-flow sharing, NOT looking for startups to invest in → Seek Networking\n"
            "   • Executive wanting peer-level exchange with others in same function, no project → Seek Networking\n"
            "   • Founder seeking peer support for mental health / founder journey / general camaraderie → Seek Networking\n"
            "IF QUESTION 4: Seek Networking.\n\n"

            "CONCRETE COUNTER-EXAMPLES — real user cases the old keyword matcher got wrong:\n"
            "   • \"Launched 3 weeks back, trying to convert first paying ones, would love to meet peer founders who cracked early GTM\" → Launch Product (NOT Seek Networking — \"meet peer founders\" is the HOW of getting GTM help for the shipped product)\n"
            "   • \"Building my professional network, open to potential co-founders and investors\" → Find Co-founder (NOT Seek Networking — cofounder intent is explicit even if softly framed)\n"
            "   • \"Looking for someone to help my company go to the next level\" → Launch Product OR Seek Mentorship (NOT Seek Networking — this is operational GTM/advisor seeking)\n"
            "   • \"Recruiting speakers for my B2B SaaS conference\" → Recruit (NOT Seek Networking — specific operational recruitment)\n"
            "   • \"Fortune 500 M&A team seeking cybersecurity founders for acquisition\" → Invest in Startups (NOT Seek Networking — M&A counts as investing)\n\n"

            # Keyword mapping preserved as fallback for cases question-1-4 don't resolve cleanly.
            "FALLBACK KEYWORD-TO-GOAL MAPPING (use only if the priority tree above didn't resolve):\n"
            "• 'co-founder', 'cofounder', 'technical partner', 'business partner', 'need someone to build', 'looking for a partner' → 'Find Co-founder'. "
            "• 'raise', 'funding', 'investors', 'seed', 'series', 'investment', 'capital' → 'Raise Funding'. "
            "• 'partnership', 'collaborate', 'strategic alliance', 'joint venture' → 'Explore Partnerships'. "
            "• 'invest', 'angel', 'deploy capital', 'fund startups' → 'Invest in Startups'. "
            "• 'job', 'career', 'employment', 'role', 'position', 'hire me' → 'Find New Job'. "
            "• 'network', 'connections', 'meet people', 'expand my network' → 'Seek Networking' (but see Question 4 above — only if no operational project). "
            "• 'hiring', 'build team', 'need talent', 'looking for developers', 'need engineers' → 'Hire Talent'. "
            "• 'recruiter', 'recruitment firm', 'staffing', 'headhunter', 'talent acquisition', 'placing candidates', 'executive search' → 'Recruit'. "
            "• 'launch', 'go-to-market', 'GTM', 'product launch', 'market entry', 'release product' → 'Launch Product'. "
            "• 'services', 'consulting', 'agency', 'freelance', 'offer expertise', 'provide services' → 'Offer Services'. "
            "IMPORTANT: A recruiter/recruitment firm is NOT a generic service provider. If someone runs a recruitment/staffing/headhunting business, use 'Recruit' not 'Offer Services'. "
            "ALWAYS extract something — never leave blank if ANY goal-related intent is mentioned."
        )
    },
    "industry_focus": {
        "description": "Industries or sectors the user focuses on",
        "type": "multi_select",
        "options": ["Technology/SaaS", "Healthcare/Biotech", "FinTech", "E-commerce", "AI/ML", "CleanTech", "EdTech", "Consumer", "Enterprise", "Other"],
        "extraction_hint": "KEYWORD-TO-INDUSTRY MAPPING (extract ALL that match): "
                          "• 'SaaS', 'software', 'platform', 'app', 'tool', 'B2B SaaS', 'API' → 'Technology/SaaS'. "
                          "• 'health', 'medical', 'hospital', 'biotech', 'pharma', 'healthcare' → 'Healthcare/Biotech'. "
                          "• 'fintech', 'financial', 'payments', 'banking', 'crypto', 'defi', 'money' → 'FinTech'. "
                          "• 'ecommerce', 'retail', 'shopping', 'marketplace', 'DTC', 'direct-to-consumer' → 'E-commerce'. "
                          "• 'AI', 'ML', 'machine learning', 'artificial intelligence', 'GPT', 'LLM' → 'AI/ML'. "
                          "• 'clean', 'green', 'sustainability', 'climate', 'energy', 'solar' → 'CleanTech'. "
                          "• 'education', 'learning', 'edtech', 'school', 'training', 'courses' → 'EdTech'. "
                          "• 'consumer', 'B2C', 'social', 'lifestyle', 'entertainment' → 'Consumer'. "
                          "• 'enterprise', 'B2B', 'corporate', 'business tools', 'remote teams', 'collaboration' → 'Enterprise'. "
                          "CRITICAL: Can select MULTIPLE industries. 'B2B SaaS' = ['Technology/SaaS', 'Enterprise']. "
                          "ALWAYS extract at least one industry if ANY business context is mentioned."
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
        "extraction_hint": "Extract mentioned regions or countries. Map specific countries to regions: London/UK → UK, Silicon Valley/US → US, etc. If user says 'fully remote', 'doesn't matter where', 'location agnostic', 'anywhere', 'geography doesn't matter', 'open to any location', 'no geography preference' → 'Global/Remote'."
    },
    "company_name": {
        "description": "Name of the user's company (if founder/executive)",
        "type": "text",
        "extraction_hint": "Extract ONLY the official registered business name - a proper noun like 'Stripe', 'MedFlow AI', 'Acme Corp'. TEST: Does it look like a brand that could be trademarked? If YES → extract. If NO → return null. EXAMPLES: 'I run a healthtech startup' → null (no name given). 'CEO of HealthTech Solutions Inc' → 'HealthTech Solutions Inc'. 'my AI company' → null. 'founder of Anthropic' → 'Anthropic'. NEVER return: 'startup', 'company', 'tech company', 'healthtech startup', 'AI platform', 'my business' - these are categories, NOT names."
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
        "description": "What the user can GIVE TO others they meet on this platform - their expertise, connections, resources",
        "type": "text",
        "extraction_hint": "OFFERINGS = What this user CAN PROVIDE to help OTHERS. Extract from their background/expertise: 'I have 20 years in X' → offering: X domain expertise; 'I know investors at Y' → offering: introductions to Y investors; 'I built companies that raised $XM' → offering: fundraising guidance. CRITICAL DISTINCTION: Offerings come from PAST EXPERIENCE, CURRENT CAPABILITIES, and NETWORK ACCESS. Do NOT confuse with what they're seeking (requirements). If user says 'I want to help startups' that's NOT an offering - they need to specify HOW they can help. CONCISE ONLY: 3-8 words max per item, semicolon-separated. Example: 'healthcare operations expertise; UCSF network introductions'"
    },
    "requirements": {
        "description": "What the user NEEDS FROM others they meet on this platform - help, connections, resources they're seeking",
        "type": "text",
        "extraction_hint": "REQUIREMENTS = What this user is ACTIVELY SEEKING from connections. Extract from stated goals/challenges: 'I need help with X' → requirement: X help/guidance; 'looking for investors' → requirement: investor introductions; 'want to expand to Europe' → requirement: European market access. CRITICAL DISTINCTION: Requirements are about GAPS TO FILL - what they DON'T have. Do NOT confuse with what they can offer. If user describes past achievements, that's offerings NOT requirements. CONCISE ONLY: 3-8 words max per item, semicolon-separated. Example: 'Series A investors; European distribution partners'. BUG-102 FIX: Do NOT extract HOW they want to engage here - engagement preferences like 'hands-on', 'collaborative', 'warm intros', 'strategic advice', 'mentorship style' go to engagement_style slot instead."
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
        "description": "Specific skills, expertise, and capabilities the user brings",
        "type": "list",
        "extraction_hint": "Extract SPECIFIC skills from all conversation turns. Be detailed, not generic. Examples: 'payment infrastructure at scale', 'distributed team management across 4 time zones', 'FDA regulatory clearance', 'React Native mobile development', 'Series A fundraising'. NEVER use just 'Technical/Engineering' — always include the specific domain."
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
        "description": "Current team size - NUMBER OF EMPLOYEES/STAFF on their own team. NOT clients, customers, or companies they serve.",
        "type": "number",
        "extraction_hint": "Extract EMPLOYEE COUNT ONLY. CRITICAL: Read carefully to distinguish employees vs customers. Example: 'leading team of 8 engineers' → 8 (team = employees). 'achieved 15 pilot customers' → DO NOT EXTRACT (customers, not employees). 'managing product roadmap for 20 clients' → DO NOT EXTRACT (clients). ONLY extract when they explicitly mention: 'team of X', 'X employees', 'X engineers on my team', 'staff of X', 'X people work for me'. NEVER extract customer/client/pilot counts. Valid range: 1-1000."
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
    },
    # BUG-075: SERVICE PROVIDER-specific slots
    "service_type": {
        "description": "Type of services offered",
        "type": "multi_select",
        "options": ["Marketing/Growth", "Engineering/Development", "Design/UX", "Legal/Compliance", "Finance/Accounting", "HR/Recruiting", "Strategy/Consulting", "Sales/BD", "Operations", "Other"],
        "extraction_hint": "Map services: 'marketing agency' → Marketing/Growth, 'dev shop' → Engineering/Development, 'legal firm' → Legal/Compliance, 'fractional CFO' → Finance/Accounting"
    },
    "budget_range": {
        "description": "Budget or pricing expectations for services",
        "type": "text",
        "extraction_hint": "Extract pricing: 'retainer $5K/month', 'project-based $10K-$50K', 'hourly $150-$300/hr', 'equity-based', 'flexible budget'"
    },
    # BUG-099: Multi-vector dimension slots - CRITICAL for match quality
    "engagement_style": {
        "description": "How the user prefers to engage with connections - their preferred relationship style",
        "options": ["Hands-on mentorship", "Strategic advice", "Introductions and network", "Purely financial", "Warm intros only", "Open to cold outreach", "Async first", "Collaborative partnership", "Monthly check-ins", "Regular syncs"],
        "extraction_hint": "BUG-102 FIX: PRIORITIZE THIS SLOT for relationship/engagement content. Extract HOW they want to work with connections: 'prefer warm intros' → 'Warm intros only'; 'happy to take cold calls' → 'Open to cold outreach'; 'want hands-on help' → 'Hands-on mentorship'; 'just need connections' → 'Introductions and network'; 'collaborative' → 'Collaborative partnership'; 'monthly check-ins' or 'regular syncs' → 'Monthly check-ins'. TRIGGER WORDS: hands-on, collaborative, strategic, mentorship, intros, check-ins, syncs, engagement style, working style, relationship. This captures HOW they want to engage, NOT what they need (requirements) or what they won't accept (dealbreakers)."
    },
    "dealbreakers": {
        "description": "What would make the user immediately pass on a connection - absolute non-starters",
        "type": "text",
        "extraction_hint": (
            "Extract ONLY items describing TYPES OF PEOPLE or SITUATIONS the user refuses to CONNECT WITH. "
            "Examples: 'not interested in generalist VCs' → 'generalist VCs'; 'won't work with pre-seed founders' → 'pre-seed companies'; 'no crypto projects' → 'crypto projects'; 'avoid consumer/marketplace deals' → 'consumer; marketplace'. "
            "CRITICAL SEMANTIC TEST — apply to every candidate phrase: "
            "'Would this user REJECT a connection because of this?' → dealbreaker. "
            "'Would this user HAPPILY connect but just not personally do this work / refer it elsewhere?' → NOT a dealbreaker (that's service scope, not avoidance). "
            "CONCRETE FAILURE CASE to prevent: a fractional CFO says 'I'll intro good lawyers or comms people when it comes up but don't pretend to do that work myself.' Lawyers/comms are services the user DOESN'T PROVIDE, not partner types the user AVOIDS. Do NOT extract 'legal' or 'comms' as dealbreakers here. The user is happy to meet founders who need legal or comms help; they just refer the work elsewhere. "
            "RULE: phrases like 'I don't do X myself', 'not my scope', 'refer out', 'hand off', 'I'll intro them to someone' → service scope, NOT dealbreakers. "
            "Phrases like 'won't work with X', 'refuse to match with X', 'no interest in X people', 'absolute no to X founders' → dealbreakers. "
            "Dealbreakers are DIFFERENT from 'requirements' — requirements = what they seek, dealbreakers = what they reject. "
            "CONCISE: 3-8 words max per item, semicolon-separated."
        )
    },
    # Conditional identity slots — activated by dependency triggers
    "achievement": {
        "description": "A key professional achievement, milestone, or result the user is proud of",
        "type": "text",
        "extraction_hint": "Extract concrete achievements: 'scaled team from 5 to 200' → 'Scaled team 5→200'; 'raised $10M Series A' → 'Raised $10M Series A'; 'grew revenue 10x' → 'Grew revenue 10x'; 'built product used by 1M users' → 'Built product reaching 1M users'. Look for numbers, growth metrics, funding milestones, exits, launches. CONCISE: 5-15 words max."
    },
    "network_strength": {
        "description": "The user's strongest professional network or community",
        "type": "text",
        "extraction_hint": "Extract their strongest connections/community: 'I know every VC in London' → 'London VC community'; 'deep network in healthcare' → 'Healthcare industry network'; 'well connected in YC alumni' → 'YC alumni network'; 'strong relationships with enterprise CTOs' → 'Enterprise CTO network'. CONCISE: 3-8 words max."
    },
    # Match pre-filter slot — WHO the user wants to connect with
    "seeking_user_types": {
        "description": "The types of people this user wants to meet on the platform — normalized to standard roles",
        "type": "multi_select",
        "options": ["Founder/Entrepreneur", "Angel Investor", "VC Partner", "Corporate Executive", "Mentor/Advisor", "Service Provider", "Recruiter", "Job Seeker/Candidate"],
        "extraction_hint": "CRITICAL: Extract WHO the user wants to CONNECT WITH, not who they ARE. "
                          "This is about the TYPE OF PERSON they are seeking. "
                          "• 'looking for founders' → ['Founder/Entrepreneur'] "
                          "• 'want to meet investors' → ['Angel Investor', 'VC Partner'] "
                          "• 'need a mentor' → ['Mentor/Advisor'] "
                          "• 'hiring engineers' → ['Job Seeker/Candidate'] "
                          "• 'looking for recruitment partners' → ['Recruiter'] "
                          "• 'need a marketing agency' → ['Service Provider'] "
                          "• 'want to connect with other investors' → ['Angel Investor', 'VC Partner'] "
                          "• If user says 'I invest in startups' with NO mention of wanting to meet investors → ['Founder/Entrepreneur'] (implied: investors seek founders) "
                          "• If user says 'I want to find a co-founder' → ['Founder/Entrepreneur'] "
                          "INFER from context if not explicitly stated: an investor looking for deal flow seeks founders. "
                          "A founder seeking funding seeks investors. A mentor seeks mentees (founders). "
                          "ALWAYS extract at least one type."
    }
}


# Semantic topic clusters - questions in same cluster are semantically equivalent
# If ANY keyword from a cluster was asked, ALL keywords in that cluster are "covered"
SEMANTIC_TOPIC_CLUSTERS = {
    "goals": ["goal", "goals", "objective", "objectives", "priority", "priorities",
              "aspiration", "aspirations", "ambition", "ambitions", "vision", "aim",
              "aims", "target", "targets", "hope", "hopes", "dream", "dreams",
              "what you want", "what do you want", "looking to achieve", "trying to achieve",
              # P4 FIX: Add creative phrasings that LLM might use
              "growth plan", "growth plans", "next chapter", "next step", "next milestone",
              "next phase", "moving forward", "path forward", "trajectory", "direction",
              "roadmap", "where you're headed", "where you see yourself", "future plans"],
    "challenges": ["challenge", "challenges", "obstacle", "obstacles", "blocker", "blockers",
                   "struggle", "struggles", "difficulty", "difficulties", "problem", "problems",
                   "pain point", "pain points", "issue", "issues", "barrier", "barriers",
                   "what's hard", "what's difficult", "keeps you up"],
    "skills": ["skill", "skills", "expertise", "experience", "strength", "strengths",
               "capability", "capabilities", "superpower", "superpowers", "good at",
               "excel at", "specialize", "specialty", "specialization", "background",
               # P4 FIX: Add variations
               "what you bring", "your edge", "your advantage"],
    "needs": ["need", "needs", "requirement", "requirements", "support", "help",
              "gap", "gaps", "looking for", "seeking", "searching for", "want from",
              "need from", "require", "missing", "lack",
              # P4 FIX: Add financial/resource keywords
              "financially", "financial", "funding", "capital", "money", "resources",
              "resource", "investment", "raise", "raising"],
    "offers": ["offer", "offers", "offering", "offerings", "provide", "bring",
               "contribute", "give", "share", "can do", "able to"],
    "geography": ["geography", "location", "region", "country", "where", "based",
                  "operate", "market", "markets", "uk", "us", "europe", "asia"],
    # BUG-036 FIX: Added "journey" keywords to prevent duplicate "company journey" questions
    "stage": ["stage", "stages", "phase", "level", "round", "seed", "series",
              "pre-seed", "growth", "early-stage", "late-stage",
              "journey", "company journey", "on this journey", "your journey",
              "where are you", "how far along"],
    "industry": ["industry", "industries", "sector", "sectors", "space", "field",
                 "domain", "vertical", "market", "niche", "focus area"],
    # BUG-034 FIX: Added target_clients cluster to prevent duplicate questions about customer type
    "target_clients": ["customer", "customers", "client", "clients", "company", "companies",
                       "organization", "organizations", "segment", "segments", "market",
                       "who you work with", "who do you work with", "types of companies",
                       "kinds of companies", "target", "targets", "serve", "serving",
                       "ideal customer", "ideal client", "customer profile", "client profile",
                       "love working with", "enjoy working with", "best clients"],
    # BUG-035 FIX: Added target_company_size cluster for company size questions
    "target_company_size": ["employees", "employee count", "team size", "company size",
                            "headcount", "staff", "50-200", "100-500", "small", "medium",
                            "enterprise", "startup", "scaleup", "smb", "mid-market"]
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
        # P0 FIX: Don't initialize client at startup - lazy load to ensure runtime access to API key
        self._client = None
        # Upgraded to Claude Sonnet 4.6 with dedicated extraction key
        from app.services.llm_fallback import ANTHROPIC_MODEL
        self.extraction_model = os.getenv("ANTHROPIC_EXTRACTION_MODEL", ANTHROPIC_MODEL)
        # Use Claude Sonnet 4.6 for personalized follow-up questions
        self.personalization_model = os.getenv("ANTHROPIC_PERSONALIZATION_MODEL", ANTHROPIC_MODEL)
        # For backwards compatibility
        self.model = self.extraction_model
        # Session-specific pattern memory (keyed by session_id to avoid cross-user pollution)
        # CRITICAL: Service is singleton, so patterns must be session-specific
        self._session_patterns = {}  # {session_id: {'openers': [], 'structures': [], 'punctuation': [], 'interpretations': []}}
        # Apr-22 Phase 2 prompt caching: cache the stable extraction rules (built once with
        # empty variable state → byte-identical across all calls, cache-control-marker goes
        # on this block). Per-turn variable state (already_filled, covered_topics, priority,
        # resume_context, BUG-045 objective filter) is formatted into the user message as
        # a SESSION STATE block. See _build_extraction_session_state and extract_slots.
        self._cached_stable_rules: Optional[str] = None

    @property
    def client(self) -> Anthropic:
        """
        P0 FIX: Lazy-load Anthropic client to ensure API key is read at runtime.

        This prevents 403 errors when the API key is accessible at runtime
        but not during service initialization.
        """
        if self._client is None:
            from app.services.llm_fallback import get_anthropic_key
            api_key = get_anthropic_key("extraction")
            if not api_key:
                raise ValueError("ANTHROPIC_EXTRACTION_KEY environment variable is required")
            logger.info(f"Initializing Anthropic client with extraction key")
            self._client = Anthropic(api_key=api_key)
        return self._client

    def _detect_user_correction(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[str]:
        """
        Detect if user is correcting a previous AI interpretation.

        Returns the last AI response if correction is detected, so we can avoid repeating it.
        """
        correction_signals = [
            "you misunderstood", "that's not what i meant", "no, i said",
            "i didn't say that", "that's wrong", "not quite", "let me clarify",
            "what i meant was", "actually, i meant", "no no", "that's incorrect",
            "you got it wrong", "i didn't mean", "misread", "misheard"
        ]

        msg_lower = user_message.lower()
        is_correction = any(signal in msg_lower for signal in correction_signals)

        if not is_correction:
            return None

        # Get the last AI response that was corrected
        if conversation_history:
            for turn in reversed(conversation_history):
                if turn.get("role") == "assistant":
                    last_ai_response = turn.get("content", "")
                    logger.warning(f"User correction detected. Last AI response: {last_ai_response[:100]}...")
                    return last_ai_response

        return None

    def _detect_opener(self, question: str) -> str:
        """Detect the opening phrase pattern of a question."""
        question = question.strip()
        # Extract first 3-5 words as the opener pattern
        words = question.split()[:5]
        opener = ' '.join(words)

        # Normalize common patterns
        if opener.startswith("That's"):
            return f"That's {words[1]}" if len(words) > 1 else "That's"
        elif opener.startswith("What a"):
            return f"What a {words[2]}" if len(words) > 2 else "What a"
        elif opener.startswith("What"):
            return "What [question]"
        elif opener.startswith("How"):
            return "How [question]"
        else:
            return ' '.join(words[:3])

    def _detect_structure(self, question: str) -> str:
        """Detect the structural pattern of a question."""
        question = question.strip()

        # Detect common structural patterns
        if "—" in question or " — " in question:
            # Has em dash
            parts = question.split("—") if "—" in question else question.split(" — ")
            if len(parts) >= 2:
                # Check if it's acknowledgment-bridge-question pattern
                if any(bridge in parts[1] for bridge in ["As you", "Since you", "When you", "Given that"]):
                    return "ack_em_dash_bridge_q"
                else:
                    return "ack_em_dash_direct_q"

        # Check for bridge phrases without em dash
        if any(bridge in question for bridge in ["As you're", "Since you're", "When you think", "Given that you"]):
            return "ack_bridge_q"

        # Direct question with parenthetical
        if "(" in question and ")" in question:
            return "direct_q_parenthetical"

        # Two-part question
        if "?" in question[:-1]:  # Question mark not at the end
            return "two_part_q"

        return "direct_q"

    def _detect_punctuation_pattern(self, question: str) -> str:
        """Detect primary punctuation pattern used."""
        if "—" in question or " — " in question:
            return "em_dash"
        elif ":" in question:
            return "colon"
        elif "(" in question:
            return "parenthetical"
        elif "," in question and question.count(",") >= 2:
            return "multi_comma"
        else:
            return "simple"

    def _repair_truncated_json(self, json_str: str) -> str:
        """
        Repair truncated JSON by closing open braces and brackets.
        Handles cases where LLM response was cut off mid-JSON.
        """
        import re

        # Count open braces/brackets
        open_braces = json_str.count("{") - json_str.count("}")
        open_brackets = json_str.count("[") - json_str.count("]")

        # If we're in a string, try to close it
        # Simple heuristic: odd number of unescaped quotes means unclosed string
        quotes = re.findall(r'(?<!\\)"', json_str)
        if len(quotes) % 2 == 1:
            json_str += '"'

        # Close any open brackets first, then braces
        json_str += "]" * max(0, open_brackets)
        json_str += "}" * max(0, open_braces)

        return json_str

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON errors:
        - Missing commas between properties
        - Trailing commas
        - Single quotes instead of double quotes
        - Unquoted keys
        """
        import re

        # Replace single quotes with double quotes (outside of already double-quoted strings)
        # This is a simple heuristic, not perfect
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)

        # Remove trailing commas before } or ]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        # Try to fix missing commas: }" followed by "key" (common pattern)
        json_str = re.sub(r'}\s*"', r'}, "', json_str)
        json_str = re.sub(r'"\s*"([^"]+)":', r'", "\1":', json_str)

        # Fix missing comma after number followed by quote
        json_str = re.sub(r'(\d)\s+"', r'\1, "', json_str)

        # Fix missing comma after true/false/null followed by quote
        json_str = re.sub(r'(true|false|null)\s+"', r'\1, "', json_str)

        return json_str

    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON object from text that may contain preamble/postamble.

        LLMs sometimes return JSON wrapped in explanatory text like:
        "Here is the extracted data:\n{...}\nLet me know if..."

        This method finds and extracts just the JSON portion.
        """
        json_start = text.find("{")
        if json_start == -1:
            raise ValueError("No JSON object found in text")

        # Find matching closing brace
        brace_count = 0
        json_end = -1
        for i, char in enumerate(text[json_start:], start=json_start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if json_end == -1:
            # JSON was truncated, try to repair
            return self._repair_truncated_json(text[json_start:])

        return text[json_start:json_end]

    def _generate_fallback_response(self, raw_text: str, covered_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        BUG-042 FIX: Generate a fallback JSON response when LLM returns non-JSON.

        This prevents ValueError from reaching Sentry when the LLM breaks out of
        JSON format (e.g., "I apologize, I should not have repeated...").

        The fallback acknowledges the situation gracefully and asks a safe question
        that's unlikely to be a duplicate.
        """
        logger.warning(f"BUG-042 FIX: Generating fallback response. LLM returned non-JSON: {raw_text[:150]}...")

        # Determine which topics to avoid
        safe_topics = ["geography", "timeline", "success_metrics", "collaboration_style"]
        if covered_topics:
            safe_topics = [t for t in safe_topics if t not in covered_topics]

        # Pick a safe fallback question
        fallback_questions = {
            "geography": "What regions or markets are you most focused on right now?",
            "timeline": "What's your ideal timeline for making meaningful connections?",
            "success_metrics": "How would you define success from being on this platform?",
            "collaboration_style": "What kind of working relationship works best for you?"
        }

        # Pick first available safe topic
        question = "What would be most helpful for you to connect with right now?"
        for topic in safe_topics:
            if topic in fallback_questions:
                question = fallback_questions[topic]
                break

        # Build fallback response
        fallback = {
            "is_off_topic": False,
            "extracted_slots": {},
            "user_type_inference": "unknown",
            "understanding_summary": "Fallback response generated due to parsing issue. Continuing conversation naturally.",
            "missing_important_slots": ["primary_goal", "requirements", "offerings"],
            "follow_up_question": question
        }

        logger.info(f"BUG-042 FIX: Generated fallback with question: {question[:50]}...")
        return fallback

    def extract_slots_from_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        BUG-043 FIX: Pre-extract slots from resume text during upload.

        This allows slots to be pre-filled BEFORE the first chat message,
        so the onboarding can skip questions about information already in the resume.

        Args:
            resume_text: Extracted text content from the resume

        Returns:
            Dict with extracted slots: {slot_name: {"value": ..., "confidence": ..., "source": "resume"}}
        """
        # BUG-096 FIX: Reduced minimum from 50 to 10 chars
        # Previously: Short resume text like "CEO at TechCorp" (15 chars) was skipped entirely
        # Now: Process anything with meaningful content - LLM can extract value from short text
        if not resume_text or len(resume_text.strip()) < 10:
            logger.info("BUG-096: Resume text too short for extraction (<10 chars), skipping")
            return {}

        # Truncate very long resumes to avoid context overflow
        truncated = resume_text[:5000] if len(resume_text) > 5000 else resume_text

        # Build specialized prompt for resume extraction
        system_prompt = """You are analyzing a resume/CV to extract professional profile information.

## Task
Extract structured slot data from this resume. Focus on factual information that's explicitly stated.

## Universal Slots (Extract for Everyone)
- user_type: Their primary role (Founder/Entrepreneur, Angel Investor, VC Partner, Corporate Executive, Mentor/Advisor, Service Provider)
- industry_focus: Industries they work in (Technology/SaaS, Healthcare/Biotech, FinTech, E-commerce, AI/ML, CleanTech, EdTech, Consumer, Enterprise, Other)
- experience_years: TOTAL years of professional experience. Calculate from earliest relevant position to present (e.g., work history spanning 2015-2026 = 11 years).
- role_title: Current or most recent job title
- company_name: Current or most recent company
- geography: Regions mentioned (UK, US, Europe, Asia, etc.)
- offerings: What they can offer based on their background (expertise, network, skills)
- requirements: What they're looking for (often in "What I'm Looking For" or "Seeking" sections)
- stage_preference: Company stages they work with (Pre-seed, Seed, Series A, B, C, Growth, etc.)

## Investor-Specific Slots (Extract if user_type indicates investor)
- check_size: Investment amount range (e.g., "$250K-$1M", "$50K-$500K")
- investment_thesis: Investment focus/thesis statement
- portfolio_size: Number of investments or portfolio companies

## Founder-Specific Slots (Extract if user_type indicates founder)
- funding_need: Amount seeking to raise (e.g., "$2M Seed round")
- company_stage: Current company stage (Idea, MVP, Revenue, Growth)
- team_size: Number of team members

## Rules
1. ONLY extract information explicitly stated in the resume
2. For requirements, look for "Looking for", "Seeking", "Want to connect with" sections
3. For offerings, summarize their key skills/expertise in 2-3 sentences
4. Extract what you can find — focus on accuracy over completeness
5. For career changers, experience_years should reflect TOTAL professional experience across all roles

## Response Format
Return valid JSON:
{
    "extracted_slots": {
        "slot_name": {
            "value": "extracted value",
            "reasoning": "where in resume this was found"
        }
    },
    "user_type_inference": "founder|investor|advisor|executive|service_provider|unknown"
}

CRITICAL: Only return the JSON object. No explanatory text."""

        try:
            logger.info(f"BUG-043: Extracting slots from resume ({len(truncated)} chars)")

            # BUG-043 FIX: Use Sonnet (personalization_model) for resume extraction
            # Rationale: Resume extraction happens ONCE at upload time, user expects wait.
            # Sonnet handles nuances better (career transitions, complex backgrounds).
            _msgs = [{"role": "user", "content": f"Extract profile information from this resume:\n\n{truncated}"}]
            try:
                response = self.client.messages.create(
                    model=self.personalization_model, max_tokens=4096, system=system_prompt,
                    messages=_msgs, temperature=0.1
                )
                result_text = response.content[0].text.strip()
            except Exception as api_err:
                from app.services.llm_fallback import fallback_from_anthropic_error
                result_text = fallback_from_anthropic_error(
                    service="extraction", error=api_err, system_prompt=system_prompt, messages=_msgs, max_tokens=4096, temperature=0.1
                )
                if not result_text:
                    raise api_err

            # Parse JSON response
            try:
                # Try direct parse
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try extracting JSON from response
                result_text = self._extract_json_from_text(result_text)
                result = json.loads(result_text)

            extracted_slots = result.get("extracted_slots", {})

            # Mark all slots as coming from resume
            for slot_name, slot_data in extracted_slots.items():
                slot_data["source"] = "resume"

            logger.info(f"BUG-043: Extracted {len(extracted_slots)} slots from resume: {list(extracted_slots.keys())}")
            return extracted_slots

        except Exception as e:
            logger.error(f"BUG-043: Failed to extract slots from resume: {e}")
            return {}

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

    def _detect_topic_from_slots(self, extracted_slots: Dict[str, Any]) -> Optional[str]:
        """
        Detect which topic was just discussed based on extracted slots.

        Returns topic name (e.g., 'industry', 'geography') or None.
        """
        slot_to_topic = {
            "primary_goal": "goals",
            "requirements": "needs",
            "offerings": "offers",
            "geography": "geography",
            "stage_preference": "stage",
            "industry_focus": "industry",
            "engagement_style": "engagement",
            "dealbreakers": "dealbreakers",
            "challenges": "challenges",
            "skills": "skills"
        }

        for slot_name in extracted_slots.keys():
            topic = slot_to_topic.get(slot_name)
            if topic:
                return topic
        return None

    def _get_missing_multi_vector_dimensions(
        self,
        filled_slots: Dict[str, Any],
        missing_slots: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get list of missing multi-vector dimensions that haven't been filled yet.

        These are CRITICAL for match quality and should be prioritized.

        Args:
            filled_slots: Slots extracted in the current turn
            missing_slots: All slots still missing (from context manager)
        """
        # If missing_slots is provided, use it as the authoritative source
        if missing_slots:
            # Filter to only multi-vector dimensions
            mv_missing = [
                slot for slot in missing_slots
                if slot in MULTI_VECTOR_DIMENSION_SLOTS
            ]
            return mv_missing

        # Fallback: check against filled_slots from current turn
        filled_slot_names = set(filled_slots.keys()) if filled_slots else set()

        missing = []
        for slot_name in MULTI_VECTOR_DIMENSION_SLOTS.keys():
            if slot_name not in filled_slot_names:
                missing.append(slot_name)

        return missing

    def _build_multi_vector_steering(
        self,
        missing_dimensions: List[str],
        topic_counts: Dict[str, int],
        last_topic: Optional[str]
    ) -> str:
        """
        Build prompt text to steer the AI toward collecting multi-vector dimensions.

        Also enforces topic dwelling prevention.
        """
        steering_text = ""

        # Check if we're dwelling on a topic
        if last_topic and topic_counts.get(last_topic, 0) >= MAX_QUESTIONS_PER_TOPIC:
            steering_text += f"""
⚠️ TOPIC DWELLING ALERT: You've asked {topic_counts[last_topic]} questions about '{last_topic}'.
STOP asking about '{last_topic}' and SWITCH to a different topic immediately.
"""

        # Add missing multi-vector dimensions with urgency
        if missing_dimensions:
            dim_questions = []
            for dim in missing_dimensions[:3]:  # Top 3 missing
                question = MULTI_VECTOR_DIMENSION_SLOTS.get(dim)
                if question:
                    dim_questions.append(f"  • {dim}: \"{question}\"")

            if dim_questions:
                steering_text += f"""
🎯 PRIORITY: MISSING MULTI-VECTOR DIMENSIONS (ask about these FIRST):
{chr(10).join(dim_questions)}

These dimensions are CRITICAL for match quality. Do NOT ask detail questions
(like "how do you handle compliance?") until you've covered these broad dimensions.
"""

        return steering_text

    def _detect_topic_from_question(self, question: str) -> Optional[str]:
        """
        Detect which topic a generated question is about.

        This is the HARD CONSTRAINT version - we analyze the generated question
        to see if it's about a blacklisted topic, regardless of what slots it extracts.
        """
        question_lower = question.lower()

        # Topic detection based on keywords in the question itself
        topic_keywords = {
            "industry": [
                "industry", "industries", "sector", "sectors", "space",
                "fintech", "healthtech", "edtech", "saas", "b2b", "b2c",
                "market segment", "vertical", "domain", "field"
            ],
            "geography": [
                "region", "regions", "location", "locations", "where",
                "market", "markets", "country", "countries", "uk", "us",
                "europe", "european", "asia", "global", "remote", "based",
                "geographic", "geography", "territory"
            ],
            "stage": [
                "stage", "stages", "series a", "series b", "seed",
                "pre-seed", "growth", "early-stage", "late-stage",
                "startup", "scale-up", "mature", "funding round"
            ],
            "engagement": [
                "relationship", "mentorship", "hands-on", "hands-off",
                "advice", "support", "involvement", "engaged", "engagement",
                "work with", "collaborate", "partnership"
            ],
            "dealbreakers": [
                "dealbreaker", "deal-breaker", "no-go", "avoid",
                "won't work", "not for me", "red flag", "non-starter",
                "absolute no", "pass on", "never"
            ],
            "goals": [
                "goal", "goals", "objective", "objectives", "looking for",
                "trying to", "want to", "aim", "priority", "priorities",
                "achieve", "accomplish"
            ]
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in question_lower for kw in keywords):
                return topic

        return None

    def _get_forced_dimension_question(
        self,
        missing_dimensions: List[str],
        blacklisted_topics: List[str],
        question_count: int = 0
    ) -> Optional[Tuple[str, str]]:
        """
        Get a forced direct question for a missing multi-vector dimension.

        This bypasses the LLM entirely - if we need engagement_style and it's
        missing, we ask directly. No more trusting the LLM to follow instructions.

        Args:
            missing_dimensions: List of missing MV dimension slot names
            blacklisted_topics: Topics we CANNOT ask about (already at limit)
            question_count: How many questions have been asked so far

        Returns:
            Tuple of (question, dimension_name) or None if no valid question
        """
        # Map slot names to topics for blacklist checking
        slot_to_topic = {
            "primary_goal": "goals",
            "industry_focus": "industry",
            "stage_preference": "stage",
            "geography": "geography",
            "engagement_style": "engagement",
            "dealbreakers": "dealbreakers"
        }

        # Natural, conversational questions for each dimension
        dimension_questions = {
            "engagement_style": [
                "What kind of relationship would be most valuable for you - hands-on mentorship, strategic advice, or something else entirely?",
                "When you think about working with someone, what style fits you best - being closely involved or more of a strategic sounding board?",
                "How do you prefer to engage - rolling up your sleeves together, or more of a high-level advisor role?"
            ],
            "dealbreakers": [
                "Is there anything that would be an immediate 'not for me'? Any red flags or non-starters?",
                "What would make you pass on an opportunity, even if everything else looked good?",
                "Any absolute no-gos for you? Things that would be dealbreakers regardless of potential?"
            ],
            "geography": [
                "Which regions or markets are you most focused on?",
                "Are you targeting specific geographies, or open to opportunities anywhere?",
                "Where in the world are you primarily looking to connect?"
            ],
            "stage_preference": [
                "What company stage are you most drawn to - early-stage building, or later-stage scaling?",
                "Do you prefer working with pre-seed explorers or Series A scale-ups?",
                "What stage of company excites you most?"
            ],
            "industry_focus": [
                "Are there specific industries or sectors where you're most focused?",
                "What spaces are you most interested in - fintech, healthtech, or something else?",
                "Which industries get you most excited?"
            ],
            "primary_goal": [
                "What's the main thing you're hoping to get out of this platform?",
                "What would make this really valuable for you?",
                "What's your primary objective here - what success looks like?"
            ]
        }

        # Prioritize dimensions not in blacklisted topics
        for dim in missing_dimensions:
            topic = slot_to_topic.get(dim)
            if topic and topic in blacklisted_topics:
                continue  # Skip blacklisted topics

            questions = dimension_questions.get(dim)
            if questions:
                # Rotate based on question count to avoid repetition
                idx = question_count % len(questions)
                return (questions[idx], dim)

        # All missing dimensions are blacklisted - this shouldn't happen
        # but fall back to first missing dimension
        if missing_dimensions:
            dim = missing_dimensions[0]
            questions = dimension_questions.get(dim)
            if questions:
                return (questions[0], dim)

        return None

    def _get_blacklisted_topics(self, topic_counts: Dict[str, int]) -> List[str]:
        """Get list of topics that have been asked about too many times."""
        return [
            topic for topic, count in topic_counts.items()
            if count >= MAX_QUESTIONS_PER_TOPIC
        ]

    def _is_duplicate_question(self, new_question: str, previous_questions: List[str], threshold: float = 0.7) -> bool:
        """
        Check if a question is too similar to any previously asked question.

        Uses simple word overlap similarity - if 70%+ of significant words match,
        it's considered a duplicate. This catches rephrased versions of the same question.

        Args:
            new_question: The newly generated question
            previous_questions: List of all previous AI questions
            threshold: Similarity threshold (0.7 = 70% word overlap)

        Returns:
            True if the question is a duplicate
        """
        if not previous_questions:
            return False

        # Normalize: lowercase, remove punctuation, split into words
        def normalize(text: str) -> set:
            import re
            words = re.findall(r'\b[a-z]+\b', text.lower())
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                         'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                         'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                         'through', 'during', 'before', 'after', 'above', 'below',
                         'between', 'under', 'again', 'further', 'then', 'once',
                         'here', 'there', 'when', 'where', 'why', 'how', 'all',
                         'each', 'few', 'more', 'most', 'other', 'some', 'such',
                         'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                         'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                         'until', 'while', 'about', 'against', 'any', 'both',
                         'i', 'you', 'your', 'me', 'my', 'we', 'our', 'they', 'their',
                         'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                         'im', 'youre', 'thats', 'its', 'lets', 'tell'}
            return {w for w in words if w not in stop_words and len(w) > 2}

        new_words = normalize(new_question)
        if not new_words:
            return False

        for prev_q in previous_questions:
            prev_words = normalize(prev_q)
            if not prev_words:
                continue

            # Calculate Jaccard similarity
            intersection = len(new_words & prev_words)
            union = len(new_words | prev_words)

            if union > 0:
                similarity = intersection / union
                if similarity >= threshold:
                    logger.warning(
                        f"DUPLICATE DETECTED: '{new_question[:50]}...' is {similarity:.0%} similar to "
                        f"'{prev_q[:50]}...'"
                    )
                    return True

        return False

    def _generate_personalized_followup(
        self,
        user_message: str,
        extracted_slots: Dict[str, Any],
        missing_slots: List[str],
        user_type: str,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[str]:
        """
        Generate a highly personalized follow-up question using Sonnet.
        This is the 'quality' step - Haiku extracted the data, Sonnet crafts the response.

        Args:
            user_message: What the user just said
            extracted_slots: What we learned from their message
            missing_slots: What we still need to know
            user_type: Inferred user type (founder, investor, etc.)
            session_id: Session ID for pattern tracking
            conversation_history: Full conversation history to avoid repeating questions

        Returns:
            Personalized follow-up question, or None on error
        """
        try:
            # Build a focused prompt for personalization only
            extracted_summary = ", ".join([
                f"{k}={v.get('value', v) if isinstance(v, dict) else v}"
                for k, v in extracted_slots.items()
            ])

            missing_focus = missing_slots[:3]  # Focus on top 3 missing

            # CRITICAL: Extract previous AI questions to prevent exact duplicates
            previous_questions = []
            if conversation_history:
                for turn in conversation_history:
                    if turn.get("role") == "assistant":
                        question = turn.get("content", "").strip()
                        if question and "?" in question:  # Only track questions
                            previous_questions.append(question)

            # Build forbidden questions list
            forbidden_questions_text = ""
            if previous_questions:
                recent_questions = previous_questions[-3:]  # Last 3 questions
                forbidden_questions_text = f"""
🚫 CRITICAL: QUESTIONS YOU ALREADY ASKED (NEVER REPEAT THESE):
{chr(10).join([f'  - "{q}"' for q in recent_questions])}

You MUST generate a COMPLETELY DIFFERENT question. Even similar phrasing is forbidden.
"""

            # Check if user is correcting us and track wrong interpretations
            corrected_response = self._detect_user_correction(user_message, conversation_history)
            wrong_interpretations_text = ""
            if corrected_response and session_id and session_id in self._session_patterns:
                # Store the wrong interpretation so we don't repeat it
                self._session_patterns[session_id]['interpretations'].append(corrected_response[:200])
                logger.info(f"[{session_id}] Recorded wrong interpretation to avoid")

            # Build wrong interpretations avoidance text
            if session_id and session_id in self._session_patterns:
                wrong_interps = self._session_patterns[session_id].get('interpretations', [])
                if wrong_interps:
                    wrong_interpretations_text = f"""
🚫 CRITICAL: INTERPRETATIONS YOU GOT WRONG (NEVER REPEAT THESE):
The user corrected your previous understanding. DO NOT make the same mistake:
{chr(10).join([f'  - WRONG: "{interp[:100]}..."' for interp in wrong_interps[-2:]])}

You MUST acknowledge the correction and show you NOW understand correctly.
"""

            # Get session-specific patterns to avoid repetition
            # Initialize session patterns if not exists
            if session_id and session_id not in self._session_patterns:
                self._session_patterns[session_id] = {
                    'openers': [],
                    'structures': [],
                    'punctuation': [],
                    'interpretations': [],  # Track AI interpretations to avoid repeating wrong ones
                    'topic_counts': {},  # Track questions per topic to prevent dwelling
                    'last_topic': None   # Last topic asked about
                }
                logger.info(f"[{session_id}] Initialized session pattern tracking")

            # Get recent patterns for this session
            if session_id and session_id in self._session_patterns:
                patterns = self._session_patterns[session_id]
                recent_openers = patterns['openers'][-2:] if patterns['openers'] else []
                recent_structures = patterns['structures'][-2:] if patterns['structures'] else []
                recent_punctuation = patterns['punctuation'][-2:] if patterns['punctuation'] else []
                logger.info(f"[{session_id}] Retrieved patterns: {len(patterns['openers'])} openers, {len(patterns['structures'])} structures")
            else:
                # Fallback to empty patterns if no session_id
                recent_openers = []
                recent_structures = []
                recent_punctuation = []
                if session_id:
                    logger.warning(f"[{session_id}] Session not found in pattern tracker")
                else:
                    logger.warning("No session_id provided for pattern tracking")

            # Build pattern avoidance instructions
            pattern_avoidance = ""
            if recent_openers:
                pattern_avoidance += f"\n🚫 ALREADY USED OPENERS (NEVER REPEAT): {', '.join(recent_openers)}"
            if recent_structures:
                pattern_avoidance += f"\n🚫 ALREADY USED STRUCTURES (MUST USE DIFFERENT): {', '.join(recent_structures)}"

            # =========================================================
            # MULTI-VECTOR DIMENSION STEERING (Critical for match quality)
            # =========================================================

            # Detect which topic we just discussed and track it
            current_topic = self._detect_topic_from_slots(extracted_slots)
            topic_counts = {}
            last_topic = None

            if session_id and session_id in self._session_patterns:
                patterns = self._session_patterns[session_id]
                topic_counts = patterns.get('topic_counts', {})
                last_topic = patterns.get('last_topic')

                # Update topic count for the current topic
                if current_topic:
                    topic_counts[current_topic] = topic_counts.get(current_topic, 0) + 1
                    patterns['topic_counts'] = topic_counts
                    patterns['last_topic'] = current_topic
                    logger.info(f"[{session_id}] Topic '{current_topic}' count: {topic_counts[current_topic]}")

            # Get missing multi-vector dimensions
            # Use missing_slots (from context manager) as authoritative source
            missing_mv_dimensions = self._get_missing_multi_vector_dimensions(
                filled_slots=extracted_slots,
                missing_slots=missing_slots
            )

            # Build multi-vector steering text
            multi_vector_steering = self._build_multi_vector_steering(
                missing_dimensions=missing_mv_dimensions,
                topic_counts=topic_counts,
                last_topic=last_topic
            )

            # =========================================================
            # PRE-LLM HARD CONSTRAINT: Force critical dimensions after Q4
            # =========================================================
            # If we've asked 4+ questions and engagement_style or dealbreakers
            # are still missing, force those questions directly (skip LLM).

            total_questions = sum(topic_counts.values())
            blacklisted_topics = self._get_blacklisted_topics(topic_counts)

            # Critical dimensions that MUST be asked if missing after Q4
            critical_missing = [
                dim for dim in missing_mv_dimensions
                if dim in ["engagement_style", "dealbreakers"]
            ]

            # BUG-095 FIX: Removed forced question injection that bypassed LLM
            # Previously: After Q4, code forced hardcoded questions for "critical" dimensions
            # Problem: Bypassed Sonnet's contextual understanding, caused awkward transitions
            # Now: Let LLM generate ALL follow-up questions - it has full context
            # The missing_mv_dimensions are still passed to LLM prompt for guidance
            if total_questions >= 4 and critical_missing:
                logger.info(
                    f"[{session_id}] BUG-095: Critical dimensions {critical_missing} still missing after Q{total_questions}, "
                    f"but letting LLM generate contextual question instead of forcing hardcoded one"
                )

            prompt = f"""Based on what this {user_type} just shared, generate ONE warm, personalized follow-up question.

USER SAID: "{user_message}"

WHAT WE LEARNED: {extracted_summary}

WHAT WE STILL NEED: {', '.join(missing_focus)}
{multi_vector_steering}{forbidden_questions_text}{wrong_interpretations_text}
CRITICAL PATTERN AVOIDANCE:{pattern_avoidance}

STRUCTURAL DIVERSITY RULES (CRITICAL):
1. NEVER follow the same sentence structure twice in a row
2. BANNED: Starting consecutive questions with "That's [adjective]"
3. BANNED: Using em dash (—) in the same position twice
4. BANNED: Repeating bridge phrases ("As you're...", "Since you're...", "When you think...")

REQUIRED VARIETY - Rotate between these structures:

Structure 1 - Acknowledgment + Direct Question:
"That's impressive. What stage are most founders at when you connect with them?"

Structure 2 - Context Setup + Question with Embedded Acknowledgment:
"You mentioned customer validation - that's rare for engineers. How do you approach it?"

Structure 3 - Direct Question + Parenthetical Acknowledgment:
"What excites you most about the Austin startup scene (given you're building remote-first)?"

Structure 4 - Two-Part Question (No Acknowledgment):
"When you think about ideal collaborators, what traits matter most? Technical depth, or something else?"

Structure 5 - Embedded Acknowledgment in Question:
"How do you balance product development and customer discovery - especially impressive at your stage?"

ENGAGEMENT RULES:
1. Reference SPECIFIC details from their message (not generic "that's interesting")
2. NEVER be robotic ("Thanks for sharing!", "Got it", "I see")
3. Make them feel heard and valued

Return ONLY the follow-up question, nothing else."""

            logger.info(f"Generating personalized follow-up with {self.personalization_model}")
            logger.info(f"Pattern avoidance: openers={recent_openers}, structures={recent_structures}")

            _msgs = [{"role": "user", "content": prompt}]
            try:
                response = self.client.messages.create(
                    model=self.personalization_model, max_tokens=150, messages=_msgs, temperature=0.6
                )
                followup = response.content[0].text.strip()
            except Exception as api_err:
                from app.services.llm_fallback import fallback_from_anthropic_error
                followup = fallback_from_anthropic_error(
                    service="extraction", error=api_err, system_prompt=None, messages=_msgs, max_tokens=150, temperature=0.6
                )
                if not followup:
                    raise api_err

            # =========================================================
            # HARD CONSTRAINT: Topic Blacklist Validation
            # =========================================================
            # The LLM may ignore steering instructions. This is the HARD check.
            # If the generated question is about a blacklisted topic, replace it.

            blacklisted_topics = self._get_blacklisted_topics(topic_counts)
            if blacklisted_topics:
                generated_topic = self._detect_topic_from_question(followup)

                if generated_topic and generated_topic in blacklisted_topics:
                    logger.warning(
                        f"[{session_id}] HARD CONSTRAINT: LLM ignored steering! "
                        f"Generated question about '{generated_topic}' which is blacklisted. "
                        f"Blacklist: {blacklisted_topics}"
                    )

                    # Get total question count for rotation
                    total_questions = sum(topic_counts.values())

                    # Force a question about a missing dimension
                    forced = self._get_forced_dimension_question(
                        missing_dimensions=missing_mv_dimensions,
                        blacklisted_topics=blacklisted_topics,
                        question_count=total_questions
                    )

                    if forced:
                        forced_question, forced_dimension = forced
                        logger.info(
                            f"[{session_id}] REPLACED with forced question for '{forced_dimension}': "
                            f"'{forced_question[:60]}...'"
                        )
                        followup = forced_question
                    else:
                        logger.warning(
                            f"[{session_id}] No valid forced question available. "
                            f"Keeping LLM question despite blacklist violation."
                        )

            # =========================================================
            # HARD CONSTRAINT #2: Exact duplicate question detection
            # =========================================================
            # Even if topic is different, we must not ask the same question text.
            # This catches cases where LLM rephrases or repeats exact questions.
            if previous_questions and self._is_duplicate_question(followup, previous_questions):
                logger.warning(
                    f"[{session_id}] HARD CONSTRAINT: Duplicate question detected! "
                    f"Question: '{followup[:60]}...' matches a previous question."
                )

                # Force a completely different question
                total_questions = sum(topic_counts.values()) if topic_counts else len(previous_questions)
                blacklisted = self._get_blacklisted_topics(topic_counts) if topic_counts else []

                forced = self._get_forced_dimension_question(
                    missing_dimensions=missing_mv_dimensions,
                    blacklisted_topics=blacklisted,
                    question_count=total_questions
                )

                if forced:
                    forced_question, forced_dimension = forced
                    logger.info(
                        f"[{session_id}] REPLACED duplicate with forced question for '{forced_dimension}': "
                        f"'{forced_question[:60]}...'"
                    )
                    followup = forced_question
                else:
                    # Fallback: ask a generic question that's definitely different
                    fallback_questions = [
                        "What would make this platform incredibly valuable for you specifically?",
                        "If you could wave a magic wand and change one thing about your current situation, what would it be?",
                        "What's the biggest gap in your network right now?",
                        "What kind of conversations are you hoping to have on this platform?"
                    ]
                    import random
                    followup = random.choice(fallback_questions)
                    logger.info(f"[{session_id}] Using fallback question to avoid duplicate: '{followup[:60]}...'")

            # Record patterns from this question to avoid in next one (session-specific)
            if session_id and session_id in self._session_patterns:
                opener = self._detect_opener(followup)
                structure = self._detect_structure(followup)
                punctuation = self._detect_punctuation_pattern(followup)

                self._session_patterns[session_id]['openers'].append(opener)
                self._session_patterns[session_id]['structures'].append(structure)
                self._session_patterns[session_id]['punctuation'].append(punctuation)

                logger.info(f"[{session_id}] Recorded patterns - opener: {opener}, structure: {structure}, punctuation: {punctuation}")
            else:
                logger.warning(f"No session_id provided, patterns not recorded")

            # Clean up any quotes
            if followup.startswith('"') and followup.endswith('"'):
                followup = followup[1:-1]

            logger.info(f"Sonnet personalized follow-up: {followup[:100]}...")
            return followup

        except Exception as e:
            logger.warning(f"Sonnet personalization failed, using Haiku result: {e}")
            return None  # Fall back to Haiku's follow-up

    def extract_slots(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        already_filled_slots: Optional[Dict[str, Any]] = None,
        target_slots: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        resume_context: Optional[str] = None,
        priority_extract_slots: Optional[List[str]] = None
    ) -> LLMExtractionResult:
        """
        Extract slot values from user message using LLM comprehension.

        Args:
            user_message: The user's latest message
            conversation_history: Previous turns for context
            already_filled_slots: Slots already extracted in this session
            target_slots: Specific slots to focus on (None = all)
            session_id: Session ID for logging
            resume_context: Optional extracted resume text to inform extraction
            priority_extract_slots: Deferred slots from earlier turns to prioritize

        Returns:
            LLMExtractionResult with extracted slots and follow-up
        """
        already_filled = already_filled_slots or {}
        history = conversation_history or []
        self._priority_extract_slots = priority_extract_slots or []

        # ISSUE-1 FIX: Log if we have resume context
        if resume_context:
            logger.info(f"ISSUE-1 FIX: Using resume context ({len(resume_context)} chars) for slot extraction")

        # BUG-092 FIX: Removed dumb substring-based completion detection
        # Previously: _user_wants_to_finish() checked if "done" was in message
        # Problem: "done Africa deals" triggered false positive, blocked extraction
        # Fix: Let Sonnet detect completion intent - it understands context
        # Completion detection now happens via LLM response field: is_completion_signal

        # BUG-002 FIX: Detect which semantic topics have already been asked
        # This prevents GPT-4o-mini from asking "goals" vs "objectives" vs "priorities"
        covered_topics = self._detect_covered_topics(history)
        if covered_topics:
            logger.info(f"Topics already covered in conversation: {covered_topics}")

        # Build the extraction prompt with covered topics and resume context
        system_prompt = self._build_system_prompt(already_filled, target_slots, covered_topics, resume_context)

        # Build conversation context (Anthropic API: system is separate, messages are user/assistant only)
        messages = []

        # Add conversation history for context
        # CRITICAL: Anthropic requires:
        # 1. Only "user" and "assistant" roles (no "system")
        # 2. First message must be "user"
        # 3. Messages must alternate (no consecutive same-role messages)
        for turn in history[-6:]:  # Last 6 turns for context
            role = turn.get("role", "user")
            content = turn.get("content", "")

            # Skip system messages - Anthropic doesn't accept them in messages array
            if role == "system":
                continue

            # Only accept user/assistant roles
            if role not in ("user", "assistant"):
                role = "user"

            # Skip empty content
            if not content or not content.strip():
                continue

            messages.append({"role": role, "content": content})

        # Ensure first message is "user" (Anthropic requirement)
        if messages and messages[0]["role"] != "user":
            # Insert a placeholder user message at the start
            messages.insert(0, {"role": "user", "content": "[Conversation continues from earlier context]"})

        # Add current message
        messages.append({
            "role": "user",
            "content": f"Extract information from this message:\n\n\"{user_message}\""
        })

        # Consolidate consecutive user messages (Anthropic doesn't allow them)
        consolidated = []
        for msg in messages:
            if consolidated and consolidated[-1]["role"] == msg["role"]:
                # Merge with previous message
                consolidated[-1]["content"] += "\n\n" + msg["content"]
            else:
                consolidated.append(msg)
        messages = consolidated

        # Apr-22 Phase 2 prompt caching: build the stable rules (byte-identical across all
        # calls; cache hits across all sessions and users site-wide) and the per-turn
        # SESSION STATE block (variable; lives in the user message). The LLM reads stable
        # rules first, then the SESSION STATE which overrides any stale info in the rules
        # (the cached rules were built with empty variable state for cache stability).
        stable_rules = self._get_cached_stable_rules()
        session_state_text = self._build_extraction_session_state(
            already_filled, target_slots, covered_topics, resume_context
        )

        # Prepend SESSION STATE to the first user message. Anthropic requires alternating
        # user/assistant — consolidation above already ensured first role is "user".
        messages_with_state = [dict(m) for m in messages]  # shallow copy
        if messages_with_state and messages_with_state[0]["role"] == "user":
            messages_with_state[0]["content"] = (
                "⚠️ SESSION STATE (overrides any stale info in the system rules — "
                "the rules list ALL slots and generic defaults for cache stability; "
                "THIS block is the authoritative state for the current turn):\n\n"
                f"{session_state_text}\n\n---\n\n{messages_with_state[0]['content']}"
            )
        else:
            # No user message in history — insert one carrying only the session state.
            messages_with_state.insert(0, {
                "role": "user",
                "content": f"⚠️ SESSION STATE (authoritative state for the current turn):\n\n{session_state_text}",
            })

        # Retry loop for JSON parsing failures
        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Calling Anthropic API with model: {self.model} (attempt {attempt + 1}/{max_retries + 1})")
                logger.info(f"Messages count: {len(messages)}, roles: {[m['role'] for m in messages]}")
                logger.debug(f"System prompt length: {len(system_prompt)} chars (stable={len(stable_rules)} chars)")

                # BUG-016 FIX: On retry, use drastically simplified JSON-only prompt
                # Strip away overwhelming conversational instructions that confuse the LLM.
                # Apr-22 Phase 2: on attempt 0, use the cached stable rules (byte-stable across
                # all calls, cross-session/cross-user cache shared). On retry attempts use the
                # minimal retry prompt as before; SESSION STATE stays on the user message for
                # both paths so retry still has real state.
                if attempt == 0:
                    retry_system = stable_rules
                    messages_for_call = messages_with_state
                else:
                    messages_for_call = messages_with_state
                    # Build minimal slots list for retry (all slots minus already filled)
                    all_slots = SLOT_DEFINITIONS.keys()
                    if target_slots:
                        all_slots = target_slots
                    remaining_slots_list = [s for s in all_slots if s not in already_filled]

                    retry_system = f"""🚨 CRITICAL: Your previous response was INVALID. You returned conversational text instead of JSON.

YOU MUST RETURN VALID JSON. NO EXCEPTIONS. NO CONVERSATIONAL TEXT.

## Task
Extract structured data from the user's message and return it as JSON.

## Slots to Extract
{', '.join(remaining_slots_list)}

## Already Collected (DO NOT ask again)
{', '.join(already_filled.keys()) if already_filled else 'None'}

## Required JSON Format
YOU MUST return EXACTLY this structure. NO other text allowed:

{{
    "is_off_topic": false,
    "extracted_slots": {{
        "slot_name": {{"value": "extracted value", "reasoning": "why"}}
    }},
    "user_type_inference": "founder|investor|advisor|unknown",
    "understanding_summary": "Brief analysis",
    "missing_important_slots": ["list", "of", "missing"],
    "follow_up_question": "Next question to ask the user"
}}

CRITICAL RULES:
1. Your response MUST start with {{ and end with }}
2. NO text before the JSON object
3. NO text after the JSON object
4. NO conversational preambles like "Based on the details..." or "Okay, let me try..."
5. If you return anything other than pure JSON, the system will FAIL

WRONG: "Based on the details you shared, here is what I extracted: {{"
WRONG: "Okay, let me try this again from the beginning. Based on..."
WRONG: "I'm afraid I don't have enough information..."
RIGHT: {{"is_off_topic": false, "extracted_slots": ...}}

Your response MUST be parseable JSON. Begin with {{ now."""

                # Use prompt caching for the large system prompt (reduces latency by ~80% on cache hit)
                _sys_cached = [{"type": "text", "text": retry_system, "cache_control": {"type": "ephemeral"}}]
                try:
                    response = self.client.messages.create(
                        model=self.model, max_tokens=4096, system=_sys_cached,
                        messages=messages_for_call, temperature=0.1
                    )
                    # Log full response metadata for debugging (including cache stats)
                    cache_creation = getattr(response.usage, 'cache_creation_input_tokens', 0)
                    cache_read = getattr(response.usage, 'cache_read_input_tokens', 0)
                    logger.info(f"Anthropic response: stop_reason={response.stop_reason}, cache_hit={cache_read > 0}, cache_tokens={cache_read or cache_creation}")
                except Exception as api_err:
                    from app.services.llm_fallback import fallback_from_anthropic_error
                    fallback_text = fallback_from_anthropic_error(
                        service="extraction", error=api_err, system_prompt=_sys_cached,
                        messages=messages_for_call, max_tokens=4096, temperature=0.1
                    )
                    if fallback_text:
                        # Create a mock response object so downstream code works
                        class _MockContent:
                            def __init__(self, text): self.text = text
                        class _MockResponse:
                            def __init__(self, text):
                                self.content = [_MockContent(text)]
                                self.stop_reason = "fallback"
                        response = _MockResponse(fallback_text)
                        logger.info(f"Using fallback response for extraction")
                    else:
                        raise api_err

                if not response.content:
                    logger.error("LLM returned response with empty content array")
                    raise ValueError("LLM returned empty content array")

                result_text = response.content[0].text
                logger.info(f"Anthropic response (first 300 chars): {result_text[:300] if result_text else 'EMPTY'}")

                if not result_text or not result_text.strip():
                    logger.error(f"Anthropic returned empty response. Stop reason: {response.stop_reason}")
                    raise ValueError("Empty response from Anthropic API")

                # Strip markdown code blocks if present (Claude often wraps JSON in ```json ... ```)
                result_text = result_text.strip()
                if result_text.startswith("```"):
                    # Remove opening fence (```json or ```)
                    result_text = result_text.split("\n", 1)[1] if "\n" in result_text else result_text[3:]
                if result_text.endswith("```"):
                    # Remove closing fence
                    result_text = result_text[:-3]
                result_text = result_text.strip()

                # Handle conversational preambles like "Got it, here's what I extracted:"
                # Find the first { and extract JSON from there
                if not result_text.startswith("{"):
                    json_start = result_text.find("{")
                    if json_start == -1:
                        # BUG-042 FIX: Instead of raising, use fallback response
                        # This handles cases like "I apologize, I should not have repeated..."
                        logger.warning(f"BUG-042 FIX: No JSON found in response, using fallback: {result_text[:200]}")
                        result_data = self._generate_fallback_response(result_text, covered_topics)
                        result = self._parse_llm_response(result_data, already_filled)
                        return result
                    # Find matching closing brace
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(result_text[json_start:], start=json_start):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    if json_end == -1:
                        # Try to repair truncated JSON by closing open braces
                        result_text = self._repair_truncated_json(result_text[json_start:])
                        logger.info(f"Attempted JSON repair for truncated response")
                    else:
                        result_text = result_text[json_start:json_end]
                        logger.info(f"Extracted JSON from preamble (chars {json_start}-{json_end})")

                # Try to parse, with repair on failure
                try:
                    result_data = json.loads(result_text)
                except json.JSONDecodeError as je:
                    logger.warning(f"JSON parse error: {je}, attempting repair...")
                    repaired = self._repair_json(result_text)
                    result_data = json.loads(repaired)
                    logger.info("JSON repair successful")

                # If we get here, parsing succeeded - process the result
                result = self._parse_llm_response(result_data, already_filled)

                # BUG-071 FIX: Removed follow_up_question generation from extraction
                # Question generation is now handled by separate LLMQuestionGenerator service
                # This provides clean separation of concerns and allows parallel execution

                return result

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                else:
                    # BUG-042 FIX: All retries exhausted, use fallback instead of raising
                    logger.warning(f"BUG-042 FIX: All retries exhausted ({e}), using fallback response")
                    fallback_data = self._generate_fallback_response(str(e), covered_topics)
                    return self._parse_llm_response(fallback_data, already_filled)

        # If we exit the loop without returning (all retries exhausted), return fallback
        logger.error(f"All {max_retries + 1} extraction attempts failed")
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
        covered_topics: Optional[List[str]] = None,
        resume_context: Optional[str] = None
    ) -> str:
        """Build the system prompt for extraction."""

        # ISSUE-1 FIX: Build resume context section
        resume_context_text = ""
        if resume_context:
            # Truncate to prevent context overflow
            truncated_resume = resume_context[:3000] if len(resume_context) > 3000 else resume_context
            resume_context_text = f"""
## 📄 USER'S RESUME (Background Context)
The user has uploaded a resume/CV. Use this background to:
1. **ACKNOWLEDGE IT FIRST** — Your follow_up_question MUST start by briefly acknowledging you've seen their background (e.g., "I see from your CV that you have experience in [X]..." or "Your background in [Y] is impressive...")
2. SKIP questions about information already clear from resume (e.g., industry, experience level)
3. Infer their OFFERINGS (what they can provide) from their background
4. Focus questions on what's MISSING (requirements, specific goals, what they're seeking)

Resume Summary:
{truncated_resume}

CRITICAL: Your follow_up_question MUST acknowledge the resume in the first sentence, then ask about what's NOT in the resume (their goals, what they're seeking, who they want to connect with).
DO NOT ask about information clearly stated in the resume above.
"""

        # Filter to target slots if specified
        slots_to_extract = SLOT_DEFINITIONS
        if target_slots:
            slots_to_extract = {k: v for k, v in SLOT_DEFINITIONS.items() if k in target_slots}

        # BUG-045 FIX: Apply objective-aware filtering based on user's primary_goal
        # This prevents extracting irrelevant slots (e.g., role_type for founders seeking funding)
        primary_goal = already_filled.get("primary_goal")
        if primary_goal:
            slots_to_extract = filter_slots_by_objective(slots_to_extract, primary_goal)
            logger.info(f"BUG-045 FIX: Filtered slots by objective '{primary_goal}', {len(slots_to_extract)} slots remaining")

        # Remove already filled slots from extraction targets
        remaining_slots = {k: v for k, v in slots_to_extract.items() if k not in already_filled}

        slot_descriptions = []
        for name, definition in remaining_slots.items():
            desc = f"- {name}: {definition['description']}"
            # BUG-064 FIX: Tell LLM about multi_select types so it returns arrays
            if definition.get('type') == 'multi_select':
                desc += f"\n  TYPE: multi_select (return as array, e.g., [\"Option1\", \"Option2\"])"
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

            # BUG-093 FIX: Softened language - aggressive wording made Sonnet overly cautious
            # Previously: "ABSOLUTELY FORBIDDEN" / "you will be penalized" scared LLM into avoiding valid topics
            # Now: Gentle guidance that lets Sonnet make intelligent decisions
            covered_topics_text = f"""
## Already Discussed Topics
We've already covered these areas, so please focus your next question on something new:

Already covered: {', '.join(covered_topics)}

Good alternative topics to explore: {', '.join(list(set(SEMANTIC_TOPIC_CLUSTERS.keys()) - set(covered_topics))[:8])}

Note: It's fine to briefly reference covered topics for context, just don't make them the main focus of your question.
"""

        # Determine which REQUIRED slots are still missing
        required_slots = ["primary_goal", "requirements", "offerings", "user_type", "industry_focus", "geography"]

        # BUG-051 FIX: Use get_onboarding_slots() from use_case_templates.py instead of hardcoded slots
        # This ensures LLM prompts match the same slots used by onboarding progress and embeddings
        primary_goal = already_filled.get("primary_goal", "").lower() if already_filled.get("primary_goal") else ""
        user_type_value = already_filled.get("user_type", "").lower() if already_filled.get("user_type") else ""

        # Determine objective: prefer primary_goal, fallback to user_type mapping
        objective = None
        if primary_goal:
            objective = primary_goal
        elif user_type_value:
            # Map user_type to objective (same as onboarding.py _get_core_slot_names)
            if any(keyword in user_type_value for keyword in ["founder", "entrepreneur", "building", "startup"]):
                objective = "fundraising"
            elif any(keyword in user_type_value for keyword in ["investor", "vc", "angel"]):
                objective = "investing"
            elif any(keyword in user_type_value for keyword in ["advisor", "mentor", "consultant"]):
                objective = "mentorship"
            elif any(keyword in user_type_value for keyword in ["hiring", "recruiter", "hr"]):
                objective = "hiring"
            elif any(keyword in user_type_value for keyword in ["partner", "alliance", "collaboration"]):
                objective = "partnership"
            elif any(keyword in user_type_value for keyword in ["cofounder", "co-founder"]):
                objective = "cofounder"
            elif any(keyword in user_type_value for keyword in ["launch", "product", "gtm"]):
                objective = "product_launch"

        # Get objective-specific focus slots from use_case_templates.py
        # BUG-088 FIX: These slots define the PRIORITY ORDER for progressive disclosure
        focus_slots = []
        if objective:
            try:
                focus_slots = get_onboarding_slots(objective)
                # Add focus slots that aren't already in required_slots
                for slot in focus_slots:
                    if slot not in required_slots:
                        required_slots.append(slot)
                logger.debug(f"Added objective-specific slots for '{objective}': {focus_slots}")
            except Exception as e:
                logger.warning(f"Could not get onboarding slots for objective '{objective}': {e}")

        # BUG-088 FIX: Build priority slots text for progressive disclosure
        # This tells the LLM the ORDER in which to extract slots (max 3 per turn)
        if focus_slots:
            # Filter to only slots not yet filled
            remaining_priority = [s for s in focus_slots if s not in already_filled]
            priority_slots_text = f"Extract in this order (first 3 that apply): {', '.join(remaining_priority[:6])}"
        else:
            # Fallback: use required_slots order
            remaining_priority = [s for s in required_slots if s not in already_filled]
            priority_slots_text = f"Extract in this order (first 3 that apply): {', '.join(remaining_priority[:6])}"

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
2. Show authentic curiosity with VARIED phrases (never repeat the same opener twice in a conversation)
3. Build rapport through warm, personalized acknowledgments before asking the next question
4. Extract information naturally through engaged conversation, not form-filling
5. NEVER repeat questions about topics already covered

## 🌟 ENGAGEMENT STYLE (CRITICAL)

**ROBOTIC (NEVER DO THIS):**
- "Thanks for sharing! What are your goals?"
- "Got it. What industry are you in?"
- "I see. What's your timeline?"

## 🎯 DIVERSE QUESTION STRUCTURES (ROTATE - NEVER USE SAME TWICE)

**Structure 1 - Acknowledgment First + Direct Question:**
Example: "That's impressive - building in productivity with 5 years of engineering expertise. What stage are most founders at when you work with them?"
When to use: After user shares significant achievement

**Structure 2 - Context Setup + Question with Embedded Acknowledgment:**
Example: "You mentioned customer validation - that's rare for engineers at your stage. How do you approach balancing product development with user research?"
When to use: When building on specific detail they mentioned

**Structure 3 - Direct Question + Parenthetical Acknowledgment:**
Example: "What excites you most about the Austin startup scene (given you're building remote-first tools)?"
When to use: When acknowledgment can be naturally embedded

**Structure 4 - Two-Part Question (No Acknowledgment Needed):**
Example: "When you think about ideal collaborators, what matters most? Deep technical expertise, or shared vision for the product?"
When to use: When exploring preferences or priorities

**Structure 5 - Embedded Acknowledgment in Question Flow:**
Example: "How do you balance technical architecture decisions with customer needs - especially impressive given you're doing both?"
When to use: When highlighting a skill while asking about process

**Structure 6 - Insight-Based Opening:**
Example: "Building remote team tools while working remotely yourself - there's great product-market fit there. How has that influenced your approach?"
When to use: When you notice an interesting connection or insight

## 🔑 PERSONALIZATION RULES

1. **Reference their specific details**: If they said "AI for hospitals", say "AI for hospitals" back to them, not "your company"
2. **Acknowledge their unique angle**: What makes THEIR story different? Highlight it.
3. **NEVER repeat sentence structures**: Each question should feel fresh and unpredictable
4. **AVOID pattern markers**: Don't start 2+ questions with "That's...", don't use "—" in same position twice
5. **Build on their narrative**: Connect their past to their present to their future
6. **Ask story-driven follow-ups**: "What led you to that decision?" not "What's your goal?"
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

## 🎯 PROGRESSIVE DISCLOSURE (CRITICAL - BUG-088 FIX)

**MAXIMUM 3 SLOTS PER TURN** — This creates natural conversation flow.

Even if user provides information for 10 slots, you MUST:
1. Extract ONLY the TOP 3 most important slots (in priority order below)
2. ACKNOWLEDGE other information naturally ("I noticed you mentioned...")
3. Leave remaining slots for follow-up questions

**PRIORITY ORDER FOR EXTRACTION:**
{priority_slots_text}

**EXAMPLE - CORRECT BEHAVIOR:**
User: "I'm a senior PM with 8 years experience, looking for $180k+ remote fintech roles. I have skills in Agile, SQL, and stakeholder management."

This message contains 6+ possible slots. Extract ONLY TOP 3:
✅ EXTRACT (top 3 priority):
- role_type: "Product Manager" (senior PM)
- seniority_level: "Senior" (senior PM)
- industry_focus: ["FinTech"] (fintech roles)

❌ DO NOT EXTRACT YET (will get in next turns):
- compensation_range: $180k+ (acknowledged but not extracted)
- remote_preference: Remote (acknowledged but not extracted)
- skills_have: Agile, SQL, stakeholder management (acknowledged but not extracted)

In understanding_summary, note: "User also mentioned $180k+ compensation, remote preference, and skills (Agile, SQL, stakeholder management) - will confirm in follow-up"

**WHY THIS MATTERS:**
- Creates natural conversation (not form-filling)
- Ensures we ASK personalized questions about remaining slots
- Builds rapport through multi-turn dialogue
- User feels heard (we acknowledge) but conversation continues

## Critical Rules
1. ALWAYS reference specific details from user's message to show you were listening (DO mirror their words with enthusiasm)
2. NEVER use generic acknowledgments like "Thanks for sharing!" or "Got it" — always personalize
3. NEVER ask for information already collected (see ALREADY COLLECTED section)
4. NO word limit - ask thoughtful questions that yield rich responses
5. If user says "looking for investors" or "raising funding" - they are a FOUNDER, not an investor
6. CEO, Founder, Co-founder = Founder/Entrepreneur ONLY if they ARE one. If they are SEEKING a CTO/co-founder/executive ROLE to join = Job Seeker/Candidate
7. PRIORITIZE questions that can fill multiple missing slots at once
8. Make them feel like the most interesting person you've talked to today

## Slots to Extract
{chr(10).join(slot_descriptions)}
{already_filled_text}{resume_context_text}
## Response Format
Return valid JSON:
{{
    "is_off_topic": false,
    "is_completion_signal": false,
    "extracted_slots": {{
        "slot_name": {{
            "value": "extracted value",
            "reasoning": "why you extracted this - include implicit inferences"
        }}
    }},
    "user_type_inference": "founder|investor|advisor|executive|service_provider|unknown",
    "understanding_summary": "INTERNAL ONLY - your analysis notes including implicit signals detected",
    "missing_important_slots": ["REQUIRED slots first: primary_goal, requirements, offerings, user_type, industry_focus, stage_preference, geography"]
}}

## 🏁 COMPLETION SIGNAL DETECTION (BUG-092 FIX)

Set "is_completion_signal": true ONLY if user EXPLICITLY wants to end onboarding:
- "I'm done" / "that's all" / "let's see my matches" / "I'm ready to start matching"
- "no more questions" / "that covers it" / "wrap it up"

Set "is_completion_signal": false for:
- Normal conversation providing information
- "done" as past tense: "I've done 10 deals" / "done Africa deals" ← NOT completion
- "finished" as past tense: "I finished my MBA" ← NOT completion
- Any message that contains substantive information to extract

CRITICAL: "done" in context like "I've done X" or "who've done Y" is PAST TENSE, not completion!

NOTE: DO NOT include follow_up_question - question generation is handled by a separate dedicated service.

## 🔧 USER CORRECTION HANDLING (CRITICAL)

If user says ANY of these correction signals:
- "you misunderstood"
- "that's not what I meant"
- "no, I said"
- "I didn't say that"
- "that's wrong"
- "not quite"
- "let me clarify"
- "what I meant was"
- "actually, I meant"

Then you MUST:
1. STOP and re-read their message carefully
2. Extract the CORRECTED meaning, not your previous interpretation
3. Acknowledge the correction naturally: "Ah, I see what you mean now"
4. Update any slots with the CORRECTED values
5. NEVER repeat the same wrong interpretation

## 🕐 TENSE AWARENESS (CRITICAL - PREVENTS MISUNDERSTANDING)

Pay close attention to VERB TENSE - it changes meaning completely:

**FUTURE/GOAL tense = What they WANT (extract as primary_goal or requirements):**
- "I want to land enterprise clients" → primary_goal: seeking enterprise clients
- "I'm trying to raise funding" → primary_goal: fundraising
- "I'm looking to expand into Europe" → requirements: European expansion support
- "hoping to", "aiming to", "planning to", "working towards"

**PAST/ACHIEVEMENT tense = What they DID (extract as offerings or experience):**
- "I landed my first enterprise client" → offerings: enterprise sales experience
- "I raised a seed round" → offerings: fundraising experience
- "I expanded into Europe" → experience: European market expertise
- "achieved", "built", "grew", "successfully", "managed to"

**EXAMPLES OF TENSE CONFUSION TO AVOID:**
❌ WRONG: User says "I'm trying to land my first enterprise client" → You extract as achievement
✅ RIGHT: User says "I'm trying to land my first enterprise client" → Extract as goal/requirement

❌ WRONG: User says "I landed 3 enterprise clients last year" → You extract as goal
✅ RIGHT: User says "I landed 3 enterprise clients last year" → Extract as achievement/offering

**COMPOUND SENTENCES - Handle both parts:**
"I landed my first client (past) and now I'm trying to scale (future)"
→ offerings: proven client acquisition
→ primary_goal: scaling/growth support

MUST NOT contain:
- "The user is..." or "They are..." (third person)
- Generic filler: "Nice to meet you!", "Thanks for sharing!", "Great!", "Got it."
- Form-like questions: "What is your X?", "Tell me about your Y"
- Questions semantically similar to previously asked ones

MUST contain:
- Specific reference to something unique from their message (names, companies, decisions, experiences)
- VARIED curiosity phrases (rotate: "fascinating", "smart move", "stands out", "impressive", "interesting") — NEVER use the same phrase twice in a row
- Natural conversational flow (not interrogation)

## Strategic Question Examples (Engaging Style) — NOTE THE VARIETY

**If user mentioned leaving a job to start something:**
"That's a brave move — walking away from stability to chase something you believe in. What was the moment you knew you had to make that leap?"

**If user mentioned a specific industry (e.g., healthtech):**
"Healthcare is such a complex space — but the potential for impact is massive. What drew you to that world specifically?"

**If user mentioned raising funding:**
"Fundraising can be quite a rollercoaster. What's been the most surprising thing about that process so far?"

**If user mentioned a co-founder search:**
"Finding the right co-founder is like dating but with higher stakes! What qualities matter most to you in that partnership?"

**If user is an investor:**
"The investor mindset is always interesting to understand. What's your thesis — what gets you genuinely excited to write a check?"

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
        "industry_focus": {{"value": ["Technology/SaaS", "Enterprise"], "reasoning": "B2B SaaS = Technology/SaaS + Enterprise"}},
        "experience_years": {{"value": "2 years", "reasoning": "Explicitly stated 'been at it for 2 years'"}},
        "team_size": {{"value": "small team", "reasoning": "Mentioned 'small team' - likely 2-10 people"}},
        "stage_preference": {{"value": ["Series A"], "reasoning": "Raising Series A — return as array because stage_preference is multi_select (user may target multiple stages)"}},
        "geography": {{"value": ["Europe"], "reasoning": "Expanding into Europe - likely current market elsewhere"}},
        "primary_goal": {{"value": "Raise Funding", "reasoning": "Looking to raise Series A"}}
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

Extract ALL inferable information. Ask questions that reveal what's STILL MISSING.

## 🚨 CRITICAL OUTPUT REQUIREMENT (NON-NEGOTIABLE)

YOU MUST RETURN VALID JSON. NO EXCEPTIONS.

**ABSOLUTELY FORBIDDEN:**
- Conversational responses without JSON
- Preambles before the JSON object
- Explanations after the JSON object
- Any text that is not valid JSON
- APOLOGIES outside of JSON (NEVER say "I apologize..." without wrapping in JSON)
- META-COMMENTARY about the conversation (NEVER say "I do not see the message..." without JSON)

**REQUIRED:**
- Your response MUST start with {{
- Your response MUST end with }}
- Your response MUST be parseable as JSON
- NO text before or after the JSON object

**BUG-016 FIX:** If you return ANY text that is not valid JSON (like "Okay, let me try this again..." or "Based on the details you shared..."), the system will FAIL. Your ONLY valid response is the JSON object defined above. Nothing else.

## 🚨 BUG-041 FIX: HANDLING CONFUSION/REPETITION (STAY IN JSON)

If you realize you're about to repeat a question or are confused about the conversation:
- NEVER break out of JSON format to apologize or explain
- ALWAYS output valid JSON with a DIFFERENT question
- Put your acknowledgment INSIDE the follow_up_question field

**EXAMPLE - IF YOU DETECT REPETITION:**
{{
    "is_off_topic": false,
    "extracted_slots": {{}},
    "user_type_inference": "unknown",
    "understanding_summary": "User may have already answered this. Switching topics.",
    "missing_important_slots": ["geography", "industry_focus"],
    "follow_up_question": "Actually, let me ask something different - what regions or markets are you most focused on?"
}}

**EXAMPLE - IF YOU'RE CONFUSED ABOUT CONVERSATION:**
{{
    "is_off_topic": false,
    "extracted_slots": {{}},
    "user_type_inference": "unknown",
    "understanding_summary": "Need to understand user's direction better.",
    "missing_important_slots": ["primary_goal", "requirements"],
    "follow_up_question": "I want to make sure I'm on the right track - what would be most helpful for you to connect with right now?"
}}

NEVER DO THIS (causes system failure):
"I apologize, I should not have repeated the same question."
"I do not see the message 'wrap up' in the conversation."

ALWAYS OUTPUT JSON, even when confused or apologizing."""

    def _get_cached_stable_rules(self) -> str:
        """Get the byte-stable extraction rules for the cached system block.

        Apr-22 Phase 2 prompt caching:
        Calls _build_system_prompt with empty variable state (already_filled={},
        target_slots=None, covered_topics=[], resume_context=None). The output is
        deterministic byte-for-byte per call — caches cleanly across all sessions
        and users site-wide. Memoized on the instance so we don't rebuild the
        ~18K-char prompt on every extraction call.

        The real per-turn variable state is moved to a SESSION STATE block in
        the user message (see _build_extraction_session_state), which the LLM
        reads as the "overrides-anything-stale-in-system-rules" source of truth
        for the current turn.
        """
        if self._cached_stable_rules is None:
            self._cached_stable_rules = self._build_system_prompt(
                already_filled={},
                target_slots=None,
                covered_topics=[],
                resume_context=None,
            )
            logger.info(
                f"Apr-22 Phase 2: cached stable extraction rules ({len(self._cached_stable_rules)} chars, "
                f"~{len(self._cached_stable_rules)//4} tok estimate)"
            )
        return self._cached_stable_rules

    def _build_extraction_session_state(
        self,
        already_filled: Dict[str, Any],
        target_slots: Optional[List[str]],
        covered_topics: Optional[List[str]],
        resume_context: Optional[str],
    ) -> str:
        """Build a SESSION STATE string for the user message (per-turn variable content).

        Apr-22 Phase 2 prompt caching:
        The stable extraction rules (cached in the system block) were built with
        empty variable state, so they contain stale defaults for things like
        "REQUIRED SLOTS STILL MISSING" (claims all 7 are missing) and priority
        order (uses default). This session-state block tells the LLM the REAL
        state for the current turn — already-filled slots, covered topics,
        priority for this turn, BUG-045 objective-filter exclusions, and resume
        context if present. The ⚠️ OVERRIDES header makes explicit that this
        block supersedes anything in the system rules that contradicts it.

        Preserves every per-turn behavior from the original _build_system_prompt:
          - BUG-045 FIX: objective-filter slot exclusions (shown as "DO NOT extract")
          - BUG-088 FIX: 3-slot-per-turn cap, priority ordering
          - BUG-002 FIX: covered topics
          - ISSUE-1 FIX: resume context
        """
        from app.services.use_case_templates import get_onboarding_slots

        parts = []

        # Resume context (session-stable, present only for users who uploaded a resume).
        if resume_context:
            truncated_resume = resume_context[:3000] if len(resume_context) > 3000 else resume_context
            parts.append(
                "## 📄 USER'S RESUME (Background Context)\n"
                "The user has uploaded a resume/CV. Use this background to:\n"
                "1. **ACKNOWLEDGE IT FIRST** — Your follow_up_question MUST start by briefly acknowledging you've seen their background (e.g., \"I see from your CV that you have experience in [X]...\" or \"Your background in [Y] is impressive...\")\n"
                "2. SKIP questions about information already clear from resume (e.g., industry, experience level)\n"
                "3. Infer their OFFERINGS (what they can provide) from their background\n"
                "4. Focus questions on what's MISSING (requirements, specific goals, what they're seeking)\n\n"
                f"Resume Summary:\n{truncated_resume}\n\n"
                "CRITICAL: Your follow_up_question MUST acknowledge the resume in the first sentence, then ask about what's NOT in the resume (their goals, what they're seeking, who they want to connect with).\n"
                "DO NOT ask about information clearly stated in the resume above."
            )

        # Already-filled slots (per-turn variable, grows as session progresses).
        if already_filled:
            filled_items = [f"- {k}: {v}" for k, v in already_filled.items()]
            parts.append(
                "## ALREADY COLLECTED - DO NOT ASK AGAIN\n"
                "The following information has ALREADY been collected. Do NOT ask for these again:\n"
                + "\n".join(filled_items)
                + "\n\nCRITICAL: Your follow_up_question must NEVER ask for information listed above. Only ask about what's MISSING."
            )

        # Covered topics (per-turn variable).
        if covered_topics:
            covered_set = set(covered_topics)
            alt_topics = [k for k in SEMANTIC_TOPIC_CLUSTERS.keys() if k not in covered_set][:8]
            parts.append(
                "## Already Discussed Topics\n"
                "We've already covered these areas, so please focus your next question on something new:\n\n"
                f"Already covered: {', '.join(covered_topics)}\n\n"
                f"Good alternative topics to explore: {', '.join(alt_topics)}\n\n"
                "Note: It's fine to briefly reference covered topics for context, just don't make them the main focus of your question."
            )

        # Objective-based slot filter (BUG-045 FIX). If primary_goal is known, compute the
        # "slots to exclude" and pass that as guidance — the LLM sees all slot definitions
        # in the cached rules but is told here which ones do NOT apply to this user.
        primary_goal = already_filled.get("primary_goal")
        slots_to_exclude_list: List[str] = []
        if primary_goal:
            full = dict(SLOT_DEFINITIONS)
            filtered = filter_slots_by_objective(full, primary_goal)
            slots_to_exclude_list = sorted(set(full.keys()) - set(filtered.keys()))
            if slots_to_exclude_list:
                parts.append(
                    f"## 🚫 DO NOT EXTRACT these slots (BUG-045 objective filter for goal '{primary_goal}')\n"
                    f"{', '.join(slots_to_exclude_list)}\n\n"
                    "These slot definitions appear in the system rules only because the rules list ALL slots for cache stability. For THIS user's objective, these slots are irrelevant. Do not emit values for them even if the conversation mentions them in passing."
                )

        # Priority ordering + missing required (BUG-088 FIX).
        required_slots = ["primary_goal", "requirements", "offerings", "user_type", "industry_focus", "geography"]
        focus_slots: List[str] = []
        user_type_value = (already_filled.get("user_type") or "").lower()
        objective: Optional[str] = None
        if primary_goal:
            objective = primary_goal
        elif user_type_value:
            if any(k in user_type_value for k in ["founder", "entrepreneur", "building", "startup"]):
                objective = "fundraising"
            elif any(k in user_type_value for k in ["investor", "vc", "angel"]):
                objective = "investing"
            elif any(k in user_type_value for k in ["advisor", "mentor", "consultant"]):
                objective = "mentorship"
            elif any(k in user_type_value for k in ["hiring", "recruiter", "hr"]):
                objective = "hiring"
            elif any(k in user_type_value for k in ["partner", "alliance", "collaboration"]):
                objective = "partnership"
            elif any(k in user_type_value for k in ["cofounder", "co-founder"]):
                objective = "cofounder"
            elif any(k in user_type_value for k in ["launch", "product", "gtm"]):
                objective = "product_launch"
        if objective:
            try:
                focus_slots = get_onboarding_slots(objective)
                for s in focus_slots:
                    if s not in required_slots:
                        required_slots.append(s)
            except Exception as e:
                logger.warning(f"Could not get onboarding slots for objective '{objective}': {e}")

        remaining_priority = [s for s in (focus_slots or required_slots) if s not in already_filled]
        priority_slots_text = f"Extract in this order (first 3 that apply): {', '.join(remaining_priority[:6])}"
        parts.append(
            "## 🎯 PRIORITY SLOTS FOR THIS TURN (BUG-088 3-slot-per-turn cap)\n"
            f"{priority_slots_text}"
        )

        missing_required = [s for s in required_slots if s not in already_filled]
        if missing_required:
            parts.append(
                "## ⚠️ REQUIRED SLOTS STILL MISSING\n"
                "These slots are REQUIRED for profile completion. Your follow-up question should naturally guide toward collecting one of these:\n"
                f"{', '.join(missing_required)}\n\n"
                "- requirements = What they NEED from connections (funding, advisors, partnerships, etc.)\n"
                "- offerings = What they can OFFER to connections (capital, expertise, introductions, etc.)\n"
                "- geography = Where they're focused (UK, US, Europe, Asia, Global, etc.)\n"
                "- stage_preference = What company stages they work with (Pre-seed, Seed, Series A, etc.)\n\n"
                "CRITICAL: If 3+ required slots are missing, prioritize collecting them over drilling into details."
            )

        # Target-slot override (rare; used when caller explicitly specifies which slots to extract).
        if target_slots:
            parts.append(
                "## 🎯 TARGET SLOTS (caller-specified focus)\n"
                f"Focus extraction on this subset: {', '.join(target_slots)}"
            )

        return "\n\n".join(parts)

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
                value = slot_data.get("value")
                # Confidence hardcoded — LLM no longer calculates it.
                # Correctness is handled by end-of-onboarding validation call.
                confidence = 0.90
                reasoning = slot_data.get("reasoning", "")
            else:
                # Handle simple value format
                value = slot_data
                confidence = 0.90
                reasoning = ""

            # BUG-013 FIX: Convert lists to strings for offerings/requirements
            # LLM sometimes returns lists like ["item1", "item2"] instead of strings
            # This causes crashes in embedding_service (.strip() on list) and DynamoDB serialization
            if slot_name in ["offerings", "requirements"] and isinstance(value, list):
                # Join list items with semicolons (as per extraction hint in prompt)
                value = "; ".join(str(item).strip() for item in value if item)
                logger.info(f"BUG-013 FIX: Converted {slot_name} from list to string: {value[:100]}...")
                # Reduce confidence slightly since we had to convert format
                confidence = max(0.7, confidence - 0.1)

            # CRITICAL VALIDATION: Sanity check for team_size to prevent "7 years" → 7000000 bug
            if slot_name == "team_size" and value is not None:
                try:
                    # Try to parse as number
                    team_num = int(str(value).replace(",", "").strip())
                    # Valid team size: 1-1000 (reasonable bounds)
                    if team_num < 1 or team_num > 1000:
                        logger.warning(f"Invalid team_size {team_num} (likely confused with years), skipping slot")
                        continue  # Skip this slot entirely - don't extract invalid data
                except (ValueError, AttributeError):
                    # Not a valid number, skip
                    logger.warning(f"team_size value '{value}' is not a valid number, skipping slot")
                    continue

            # CRITICAL VALIDATION: Reject canonical placeholder values for company_name.
            #
            # History: this block previously contained a 24-pattern substring-match
            # "generic descriptions" list ("startup", "company", "a ", "an ", "ai ",
            # "tech", etc.). That was a [[CODING-DISCIPLINE]] Rule 5 violation
            # ("Semantic LLM, Not Keyword LLM") — the set of "generic company
            # descriptions" is open-ended English, not a closed enum. Worse, the
            # substring match created false positives on any legitimate 2-word
            # brand where the first word ends in 'a' (pattern "a " matched
            # "Lumina AI", "Nvidia Corp", "Tesla Inc", "Figma Inc") or contained
            # "Tech" ("Pixelette Tech"). [[Apr-22]] Follow-up 12 Aaron Smith test
            # surfaced this on "Lumina AI" — entire class of AI-named startups
            # silently dropped.
            #
            # Fix ([[Apr-23]]): trust the LLM prompt at line 234 which already
            # enumerates exactly what counts as a valid company_name ("proper
            # noun like 'Stripe', 'MedFlow AI', 'Acme Corp'" + "NEVER return:
            # 'startup', 'company', 'tech company', ..."). The LLM (Sonnet 4.6)
            # is a semantic reasoner; asking it in the prompt is the right
            # enforcement layer. Keep only exact-match rejection for the handful
            # of canonical placeholder values the LLM sometimes emits literally
            # ("n/a", "none", "unknown", "tbd", "not specified"). That set IS
            # a closed enum — Rule 5 exception applies.
            if slot_name == "company_name" and value is not None and isinstance(value, str):
                value_lower = value.lower().strip()
                PLACEHOLDER_VALUES = {"n/a", "none", "unknown", "tbd", "not specified", ""}

                if value_lower in PLACEHOLDER_VALUES:
                    logger.info(
                        f"company_name {value!r} is a placeholder value. Skipping slot."
                    )
                    continue  # Skip this slot - will trigger follow-up question

                # Minimum viable length check (a legitimate brand name is at least 2 chars).
                if len(value.strip()) < 2:
                    logger.info(
                        f"company_name {value!r} too short (<2 chars). Skipping slot."
                    )
                    continue

            # BUG-005 FIX: Validate offerings/requirements are concise, not full message text
            # If LLM returns full sentences/paragraphs, truncate or flag for re-extraction
            if slot_name in ["offerings", "requirements"] and value is not None and isinstance(value, str):
                # Check if value is suspiciously long (> 500 chars suggests full message text copy-paste)
                # 150-500 chars is reasonable for detailed offerings/requirements
                if len(value) > 500:
                    logger.warning(
                        f"Extracted {slot_name} is too long ({len(value)} chars), likely full message text. "
                        f"Truncating to first 500 chars. Original: '{value[:100]}...'"
                    )
                    # Truncate at sentence boundary if possible
                    sentences = value.split('.')
                    if len(sentences) > 1 and len(sentences[0]) < 500:
                        value = sentences[0].strip()
                    else:
                        # Hard truncate at 500 chars
                        value = value[:500].strip()
                    # Reduce confidence since we had to intervene
                    confidence = max(0.5, confidence - 0.2)

            extracted_slots[slot_name] = LLMExtractedSlot(
                name=slot_name,
                value=value,
                confidence=confidence,
                reasoning=reasoning
            )

        # BUG-071 FIX: follow_up_question is no longer extracted here
        # Question generation is handled by separate LLMQuestionGenerator service

        # =========================================================================
        # BUG-090 FIX: FORCE primary_goal AS MANDATORY FIRST EXTRACTION
        # =========================================================================
        # Without primary_goal, we can't activate objective-specific focus slots.
        # This is the most critical slot - MUST be extracted before others.
        # If LLM didn't extract it, infer from message content or extracted slots.
        # =========================================================================
        primary_goal_known = "primary_goal" in already_filled or "primary_goal" in extracted_slots

        if not primary_goal_known:
            inferred_goal = None

            # Method 1: Infer from extracted slot types
            hiring_slots = {"role_type", "seniority_level", "remote_preference", "compensation_range", "hiring_timeline"}
            cofounder_slots = {"skills_have", "skills_need", "commitment_level", "equity_expectations"}
            job_search_slots = {"target_role", "desired_seniority", "salary_expectation", "work_preference", "availability"}
            investing_slots = {"check_size", "portfolio_size", "investment_thesis"}
            fundraising_slots = {"funding_need", "funding_range"}
            mentorship_slots = {"mentorship_areas", "mentorship_format", "mentorship_commitment"}

            extracted_keys = set(extracted_slots.keys())

            if extracted_keys & hiring_slots:
                inferred_goal = "HIRING"
            elif extracted_keys & cofounder_slots:
                inferred_goal = "COFOUNDER"
            elif extracted_keys & job_search_slots:
                inferred_goal = "JOB_SEARCH"
            elif extracted_keys & investing_slots:
                inferred_goal = "INVESTING"
            elif extracted_keys & fundraising_slots:
                inferred_goal = "FUNDRAISING"
            elif extracted_keys & mentorship_slots:
                inferred_goal = "MENTORSHIP"

            # BUG-094 FIX: Removed keyword-based inference (Method 2)
            # Previously: Substring matching like 'if "hire" in message_lower' caused false positives
            # Same bug class as BUG-092 ("done" in "done Africa deals")
            # Example: "I hired a great team" would incorrectly infer HIRING goal
            # Now: Only use Method 1 (slot-based inference) - let LLM extract primary_goal properly
            # The LLM prompt already asks for primary_goal extraction with full context understanding

            if inferred_goal:
                # Add primary_goal to extracted_slots
                # BUG-091 FIX: Use LLMExtractedSlot (not ExtractedSlot) with required 'name' param
                extracted_slots["primary_goal"] = LLMExtractedSlot(
                    name="primary_goal",
                    value=inferred_goal,
                    confidence=0.90,
                    reasoning=f"BUG-090: Inferred from message content and extracted slots"
                )
                logger.info(f"[BUG-090] Inferred primary_goal: {inferred_goal}")
        # =========================================================================
        # END BUG-090 FIX
        # =========================================================================

        # =========================================================================
        # BUG-088 FIX: ENFORCE MAX 2 SLOTS PER TURN (CODE-LEVEL, NOT PROMPT)
        # =========================================================================
        # The LLM ignores prompt instructions to limit extraction. We MUST enforce
        # this in code to guarantee progressive disclosure and natural conversation.
        # Increased from 2 to 3: with identity + conditional slots added,
        # we need to capture more per turn to keep onboarding short.
        # =========================================================================
        MAX_SLOTS_PER_TURN = 3

        if len(extracted_slots) > MAX_SLOTS_PER_TURN:
            # Determine objective from already_filled OR just-extracted primary_goal
            objective = None
            primary_goal = already_filled.get("primary_goal", "")

            # Check if primary_goal was just extracted
            if "primary_goal" in extracted_slots:
                primary_goal = extracted_slots["primary_goal"].value

            # Also check user_type for objective inference
            user_type = already_filled.get("user_type", "")
            if "user_type" in extracted_slots:
                user_type = extracted_slots["user_type"].value

            # Map to objective (same logic as _build_system_prompt)
            if primary_goal:
                goal_lower = str(primary_goal).lower()
                if "co-founder" in goal_lower or "cofounder" in goal_lower:
                    objective = "cofounder"
                elif "fund" in goal_lower or "raise" in goal_lower or "invest" in goal_lower:
                    objective = "fundraising"
                elif "job" in goal_lower or "career" in goal_lower or "role" in goal_lower:
                    objective = "job_search"
                elif "service" in goal_lower or "consult" in goal_lower:
                    objective = "services"
                elif "mentor" in goal_lower:
                    objective = "mentorship"
                elif "partner" in goal_lower:
                    objective = "partnership"
                elif "hire" in goal_lower or "recruit" in goal_lower or "talent" in goal_lower:
                    objective = "hiring"

            # Fallback to user_type mapping
            if not objective and user_type:
                type_lower = str(user_type).lower()
                if "founder" in type_lower or "entrepreneur" in type_lower:
                    objective = "fundraising"
                elif "investor" in type_lower:
                    objective = "investing"
                elif "job" in type_lower or "candidate" in type_lower:
                    objective = "job_search"
                elif "service" in type_lower:
                    objective = "services"

            # Get priority order for this objective
            try:
                priority_slots = get_onboarding_slots(objective) if objective else []
            except Exception:
                priority_slots = []

            # Default priority if no objective-specific order
            if not priority_slots:
                priority_slots = [
                    "primary_goal", "user_type", "industry_focus", "geography",
                    "role_type", "seniority_level", "company_stage", "stage_preference",
                    "funding_need", "check_size", "skills_have", "skills_need"
                ]

            # BUG-090 FIX: ALWAYS prioritize primary_goal if not already filled
            # Even with objective-specific slots, primary_goal must come first
            # FIX (Mar 30, 2026): Also prioritize user_type — critical for mentor/mentee
            # disambiguation. Without it, "Seek Mentorship" can't be distinguished.
            # FIX (Apr 19, 2026 — Brian Limba test): Also prioritize seeking_user_types.
            # Without it the cheap filter can't narrow to the user's target partner type
            # (e.g. founder raising Series A wanting "lead VC partners" gets flattened to
            # any reciprocal goal — 6:15 VC:Angel dilution visible in Brian's 14 matches).
            # seeking_user_types drives the cheap filter pool composition just as much as
            # primary_goal drives the reciprocity matrix.
            # F/u 38 #7 + #8 (Apr-20 — Jerry Lawler native-APK test): add company_name
            # and experience_years to forced_first. Jerry mentioned "Piperflow" turn 1 +
            # turn 6 but company_name was deferred by BUG-088 every turn and never recovered.
            # Same with "17 months in" + "prior fintech ops role" → experience_years never
            # extracted. Both slots carry signal the scoring LLM + Phase 2 explanation LLM
            # use for coherence — skipping them erodes match-explanation quality. Same
            # pattern as Apr-19 F/u 29 Fix #3 for seeking_user_types.
            forced_first = []
            if "primary_goal" not in already_filled:
                forced_first.append("primary_goal")
            if "user_type" not in already_filled:
                forced_first.append("user_type")
            if "seeking_user_types" not in already_filled:
                forced_first.append("seeking_user_types")
            if "company_name" not in already_filled:
                forced_first.append("company_name")
            if "experience_years" not in already_filled:
                forced_first.append("experience_years")

            if forced_first:
                for slot in forced_first:
                    if slot in priority_slots:
                        priority_slots.remove(slot)
                priority_slots = forced_first + list(priority_slots)
                logger.info(f"[BUG-090] Prioritizing forced slots: {forced_first}")

            # Select top 3 slots by priority order
            # BUG-088 FIX: Previously deferred slots get priority and COUNT toward the limit
            # (Apr-18 Follow-up 25: earlier implementation allowed zero-pass to bypass the cap
            #  entirely, meaning a rich first message with 16 deferred slots from a prior
            #  round would blow through the 3-slot cap to 19 slots per turn.)
            limited_slots = {}
            slots_added = 0
            acknowledged_slots = []

            # Zero pass: recover previously deferred slots first — they have priority,
            # but they DO count toward MAX_SLOTS_PER_TURN like any other slot.
            deferred_priority = getattr(self, '_priority_extract_slots', [])
            for deferred_slot in deferred_priority:
                if deferred_slot in extracted_slots:
                    if slots_added < MAX_SLOTS_PER_TURN:
                        limited_slots[deferred_slot] = extracted_slots[deferred_slot]
                        slots_added += 1
                        logger.info(f"[BUG-088] Recovered deferred slot: {deferred_slot}")
                    else:
                        # Still deferred — gets priority on the next turn
                        if deferred_slot not in acknowledged_slots:
                            acknowledged_slots.append(deferred_slot)

            # First pass: add slots that are in priority order
            for priority_slot in priority_slots:
                if priority_slot in extracted_slots and priority_slot not in limited_slots and slots_added < MAX_SLOTS_PER_TURN:
                    limited_slots[priority_slot] = extracted_slots[priority_slot]
                    slots_added += 1
                elif priority_slot in extracted_slots and priority_slot not in limited_slots:
                    acknowledged_slots.append(priority_slot)

            # Second pass: add remaining slots if we haven't hit limit
            for slot_name in extracted_slots:
                if slot_name not in limited_slots:
                    if slots_added < MAX_SLOTS_PER_TURN:
                        limited_slots[slot_name] = extracted_slots[slot_name]
                        slots_added += 1
                    else:
                        if slot_name not in acknowledged_slots:
                            acknowledged_slots.append(slot_name)

            logger.info(
                f"[BUG-088] Limited extraction from {len(extracted_slots)} to {len(limited_slots)} slots. "
                f"Extracted: {list(limited_slots.keys())}. "
                f"Acknowledged for next turn: {acknowledged_slots}"
            )

            # Update understanding_summary to note acknowledged info
            original_summary = response_data.get("understanding_summary", "")
            if acknowledged_slots:
                ack_note = f" [Acknowledged but deferred: {', '.join(acknowledged_slots)}]"
                response_data["understanding_summary"] = original_summary + ack_note

            extracted_slots = limited_slots
        # =========================================================================
        # END BUG-088 FIX
        # =========================================================================

        # BUG-092 FIX: Parse completion signal from LLM (replaces dumb substring matching)
        is_completion = response_data.get("is_completion_signal", False)
        if is_completion:
            logger.info(f"[BUG-092] LLM detected completion signal (user wants to finish)")

        return LLMExtractionResult(
            extracted_slots=extracted_slots,
            user_type_inference=response_data.get("user_type_inference", "unknown"),
            missing_slots=response_data.get("missing_important_slots", []),
            understanding_summary=response_data.get("understanding_summary", ""),
            is_off_topic=response_data.get("is_off_topic", False),
            is_completion_signal=is_completion,  # BUG-092: LLM decides, not regex
            acknowledged_slots=acknowledged_slots if 'acknowledged_slots' in locals() else []
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

        BUG-101 FIX: Use word boundary regex instead of substring matching.
        Previously: "done" in "I've done diligence" → TRUE (false positive)
        Now: Uses regex word boundaries to match whole phrases only.
        Same fix as BUG-092, BUG-099, and BUG-100.

        Args:
            message: User's message
            already_filled: Already collected slots (need minimum for valid profile)

        Returns:
            True if user wants to finish and has enough data
        """
        import re

        # Completion phrases that indicate user wants to finish
        # BUG-037 FIX: Added "wrap up" and variations to prevent wrap-up loop
        # BUG-101: These are now matched with word boundaries
        completion_phrases = [
            "done", "that's all", "that's everything", "i'm done", "im done",
            "show me my matches", "find my matches", "i'm ready", "im ready",
            "start matching", "no more", "nothing else", "finish", "complete",
            "let's start", "lets start", "proceed", "move on", "that covers",
            "ready to match", "see my matches",
            # BUG-037 FIX: Added wrap up variations
            "wrap up", "wrap", "move ahead", "we can move", "can move ahead",
            "move forward", "let's move", "lets move", "good to go", "all set"
        ]

        msg_lower = message.lower().strip()

        # BUG-101 FIX: Use word boundary regex instead of substring 'in'
        has_completion_signal = False
        for phrase in completion_phrases:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, msg_lower):
                logger.info(f"BUG-101: Completion phrase '{phrase}' matched in user message")
                has_completion_signal = True
                break

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
