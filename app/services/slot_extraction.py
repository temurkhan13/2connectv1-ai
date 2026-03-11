"""
Slot Extraction Service for Conversational Onboarding.

Extracts structured data (slots) from natural language user responses.
Each slot represents a piece of information needed to build the user's profile.

Key features:
- Slot schema definition with validation rules
- LLM-based extraction from free-form text
- Confidence scoring for extracted values
- Multi-value slot support
- Dependency tracking between slots
"""
import os
import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class SlotType(str, Enum):
    """Types of slots that can be extracted."""
    SINGLE_SELECT = "single_select"    # One value from a list
    MULTI_SELECT = "multi_select"      # Multiple values from a list
    FREE_TEXT = "free_text"            # Open-ended text
    NUMBER = "number"                  # Numeric value
    RANGE = "range"                    # Numeric range (min-max)
    BOOLEAN = "boolean"                # Yes/No
    DATE = "date"                      # Date value


class SlotStatus(str, Enum):
    """Status of a slot in the conversation."""
    EMPTY = "empty"                    # Not yet asked/filled
    PARTIAL = "partial"                # Some values extracted, may need more
    FILLED = "filled"                  # Slot is complete
    CONFIRMED = "confirmed"            # User has confirmed the value
    SKIPPED = "skipped"                # User chose to skip


@dataclass
class SlotDefinition:
    """Definition of a slot to be extracted."""
    name: str
    display_name: str
    slot_type: SlotType
    description: str
    required: bool = True
    options: List[str] = field(default_factory=list)  # For select types
    validation_pattern: Optional[str] = None          # Regex for validation
    min_value: Optional[float] = None                 # For number/range
    max_value: Optional[float] = None                 # For number/range
    depends_on: List[str] = field(default_factory=list)  # Slot dependencies
    extract_keywords: List[str] = field(default_factory=list)  # Keywords for extraction
    default_value: Any = None


@dataclass
class ExtractedSlot:
    """Result of slot extraction."""
    name: str
    value: Any
    confidence: float  # 0.0 to 1.0
    status: SlotStatus
    source_text: str   # Original text the value was extracted from
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    alternatives: List[Tuple[Any, float]] = field(default_factory=list)  # Other possible values


class SlotSchema:
    """
    Schema defining all slots for onboarding.

    This defines what information we need to collect from users.
    """

    # Core slots for all users
    CORE_SLOTS = [
        SlotDefinition(
            name="primary_goal",
            display_name="Primary Goal",
            slot_type=SlotType.SINGLE_SELECT,
            description="What the user is primarily looking for",
            required=True,
            options=[
                "Seeking Investment",
                "Looking to Invest",
                "Finding Co-founder",
                "Seeking Advisor/Mentor",
                "Offering Advisory Services",
                "Business Partnership",
                "Hiring Talent",
                "Networking"
            ],
            extract_keywords=["invest", "funding", "capital", "co-founder", "partner", "mentor", "advisor", "network", "hire", "hiring", "recruit"]
        ),
        # CRITICAL: These slots are required for embedding generation and matching
        SlotDefinition(
            name="requirements",
            display_name="What You Need",
            slot_type=SlotType.FREE_TEXT,
            description="What the user needs from connections (funding, advisors, talent, partnerships, etc.)",
            required=True,
            extract_keywords=["need", "looking for", "seeking", "want", "require", "find", "help with", "searching for", "interested in"]
        ),
        SlotDefinition(
            name="offerings",
            display_name="What You Offer",
            slot_type=SlotType.FREE_TEXT,
            description="What the user can offer to connections (capital, mentorship, introductions, expertise, etc.)",
            required=True,
            extract_keywords=["offer", "provide", "bring", "give", "contribute", "help with", "can do", "experience in", "expertise", "invest", "mentor"]
        ),
        SlotDefinition(
            name="user_type",
            display_name="User Type",
            slot_type=SlotType.SINGLE_SELECT,
            description="The user's primary role",
            required=True,
            options=[
                "Founder/Entrepreneur",
                "Angel Investor",
                "VC Partner",
                "Corporate Investor",
                "Advisor/Consultant",
                "Hiring Manager/Recruiter",
                "Job Seeker/Candidate",
                "Industry Professional"
            ],
            extract_keywords=["founder", "entrepreneur", "investor", "angel", "vc", "venture", "advisor", "consultant", "recruiter", "hiring", "hr", "candidate", "job"]
        ),
        SlotDefinition(
            name="industry_focus",
            display_name="Industry Focus",
            slot_type=SlotType.MULTI_SELECT,
            description="Industries the user is interested in",
            required=True,
            options=[
                "Fintech", "Healthtech", "Edtech", "SaaS", "B2B", "B2C",
                "E-commerce", "AI/ML", "Blockchain/Web3", "Clean Tech",
                "Real Estate", "Consumer", "Enterprise", "Deep Tech",
                "Marketplace", "Hardware", "Biotech", "Gaming"
            ],
            extract_keywords=["fintech", "health", "edu", "saas", "b2b", "b2c", "ai", "ml", "blockchain", "crypto"]
        ),
        SlotDefinition(
            name="stage_preference",
            display_name="Stage Preference",
            slot_type=SlotType.MULTI_SELECT,
            description="Investment or company stage",
            required=False,  # ISSUE #6 FIX: Role-specific versions in INVESTOR_SLOTS/FOUNDER_SLOTS
            options=[
                "Pre-seed", "Seed", "Series A", "Series B", "Series C+",
                "Growth", "Late Stage", "Any Stage"
            ],
            extract_keywords=["pre-seed", "seed", "series", "growth", "early", "late"]
        ),
        SlotDefinition(
            name="geography",
            display_name="Geographic Focus",
            slot_type=SlotType.MULTI_SELECT,
            description="Geographic regions of interest",
            required=True,
            options=[
                "UK", "US", "Europe", "Asia", "Middle East",
                "Latin America", "Africa", "Global/Remote"
            ],
            extract_keywords=["uk", "us", "europe", "asia", "global", "remote", "london", "silicon valley"]
        ),
    ]

    # Investor-specific slots
    INVESTOR_SLOTS = [
        SlotDefinition(
            name="stage_preference",
            display_name="Investment Stage Preference",
            slot_type=SlotType.MULTI_SELECT,
            description="What company stages do you invest in?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Pre-seed", "Seed", "Series A", "Series B", "Series C+",
                "Growth", "Late Stage", "Any Stage"
            ],
            extract_keywords=["pre-seed", "seed", "series", "growth", "early", "late", "stage"]
        ),
        SlotDefinition(
            name="check_size",
            display_name="Check Size",
            slot_type=SlotType.RANGE,
            description="Investment amount range",
            required=True,
            depends_on=["user_type"],
            min_value=0,
            max_value=100000000,
            extract_keywords=["invest", "check", "ticket", "range", "amount", "$", "£", "k", "m"]
        ),
        SlotDefinition(
            name="portfolio_size",
            display_name="Portfolio Size",
            slot_type=SlotType.NUMBER,
            description="Number of current investments",
            required=False,
            depends_on=["user_type"],
            extract_keywords=["portfolio", "investments", "companies", "startups"]
        ),
        SlotDefinition(
            name="investment_thesis",
            display_name="Investment Thesis",
            slot_type=SlotType.FREE_TEXT,
            description="Investment philosophy and focus areas",
            required=True,
            depends_on=["user_type"],
            extract_keywords=["thesis", "philosophy", "focus", "believe", "look for"]
        ),
    ]

    # Founder-specific slots
    FOUNDER_SLOTS = [
        SlotDefinition(
            name="stage_preference",
            display_name="Investor Stage Preference",
            slot_type=SlotType.MULTI_SELECT,
            description="What stage investors are you looking to connect with?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Pre-seed", "Seed", "Series A", "Series B", "Series C+",
                "Growth", "Any Stage"
            ],
            extract_keywords=["pre-seed", "seed", "series", "growth", "stage", "investor"]
        ),
        SlotDefinition(
            name="company_stage",
            display_name="Company Stage",
            slot_type=SlotType.SINGLE_SELECT,
            description="Current stage of the startup",
            required=True,
            depends_on=["user_type"],
            options=["Idea", "MVP", "Product-Market Fit", "Scaling", "Established"],
            extract_keywords=["idea", "mvp", "product", "market fit", "scaling", "revenue"]
        ),
        SlotDefinition(
            name="funding_need",
            display_name="Funding Need",
            slot_type=SlotType.RANGE,
            description="Amount of funding being raised",
            required=True,
            depends_on=["user_type"],
            min_value=0,
            max_value=100000000,
            extract_keywords=["raise", "raising", "need", "looking for", "seeking", "$", "£"]
        ),
        SlotDefinition(
            name="team_size",
            display_name="Team Size",
            slot_type=SlotType.NUMBER,
            description="Number of employees/staff on their team (NOT clients, customers, or companies served)",
            required=False,
            depends_on=["user_type"],
            extract_keywords=["team", "employees", "people", "members", "staff", "headcount"]
        ),
    ]

    # Common optional slots
    OPTIONAL_SLOTS = [
        SlotDefinition(
            name="engagement_style",
            display_name="Engagement Style",
            slot_type=SlotType.SINGLE_SELECT,
            description="Preferred way of working together",
            required=False,
            options=[
                "Hands-on/Active",
                "Strategic Guidance Only",
                "Board Seat",
                "Passive/Silent",
                "Flexible"
            ],
            extract_keywords=["hands-on", "active", "passive", "board", "strategic", "involved"]
        ),
        SlotDefinition(
            name="dealbreakers",
            display_name="Dealbreakers",
            slot_type=SlotType.MULTI_SELECT,
            description="Things that would disqualify a match",
            required=False,
            options=[],  # Free-form
            extract_keywords=["never", "avoid", "not interested", "dealbreaker", "exclude", "won't"]
        ),
        SlotDefinition(
            name="experience_years",
            display_name="Years of Experience",
            slot_type=SlotType.NUMBER,
            description="Years of relevant experience",
            required=False,
            min_value=0,
            max_value=50,
            extract_keywords=["years", "experience", "been doing", "since"]
        ),
        # BUG-035 FIX: Added target_company_size for consultants/service providers
        # This captures the SIZE of companies they want to work with (their clients)
        # Different from team_size which is their OWN company's employee count
        SlotDefinition(
            name="target_company_size",
            display_name="Target Company Size",
            slot_type=SlotType.SINGLE_SELECT,
            description="Size of companies you want to work with (by employee count). NOT your own team size.",
            required=False,
            options=[
                "1-10 employees (Micro)",
                "11-50 employees (Small)",
                "51-200 employees (Medium)",
                "201-500 employees (Mid-market)",
                "501-1000 employees (Large)",
                "1000+ employees (Enterprise)"
            ],
            extract_keywords=["employees", "company size", "companies with", "50-200", "100-500",
                            "startups", "scaleups", "enterprises", "smb", "mid-market", "headcount"]
        ),
    ]

    # ========================================================================
    # HIRING-specific slots (for companies seeking talent)
    # ========================================================================
    HIRING_SLOTS = [
        SlotDefinition(
            name="role_type",
            display_name="Role Type",
            slot_type=SlotType.SINGLE_SELECT,
            description="What type of role are you hiring for?",
            required=True,
            depends_on=["primary_goal"],
            options=[
                "Engineering/Technical",
                "Product Management",
                "Sales/Business Development",
                "Marketing/Growth",
                "Operations",
                "Executive/C-Suite",
                "Finance/Legal",
                "Other"
            ],
            extract_keywords=["engineer", "developer", "product", "sales", "marketing", "growth", "operations", "cto", "ceo", "cfo", "vp"]
        ),
        SlotDefinition(
            name="seniority_level",
            display_name="Seniority Level",
            slot_type=SlotType.SINGLE_SELECT,
            description="What level of seniority?",
            required=True,
            depends_on=["primary_goal"],
            options=[
                "Junior/Entry-level",
                "Mid-level",
                "Senior",
                "Lead/Staff",
                "Director",
                "VP/Executive",
                "C-Suite"
            ],
            extract_keywords=["junior", "mid", "senior", "lead", "staff", "director", "vp", "chief", "head of", "principal"]
        ),
        SlotDefinition(
            name="remote_preference",
            display_name="Remote Preference",
            slot_type=SlotType.SINGLE_SELECT,
            description="Work location preference",
            required=False,
            depends_on=["primary_goal"],
            options=[
                "Fully Remote",
                "Hybrid",
                "On-site Only",
                "Flexible"
            ],
            extract_keywords=["remote", "hybrid", "onsite", "on-site", "office", "flexible", "wfh"]
        ),
        SlotDefinition(
            name="compensation_range",
            display_name="Compensation Range",
            slot_type=SlotType.RANGE,
            description="Budget/expectations for the role",
            required=False,
            depends_on=["primary_goal"],
            min_value=0,
            max_value=10000000,
            extract_keywords=["salary", "compensation", "pay", "budget", "$", "£", "k", "tc", "total comp"]
        ),
    ]

    # ========================================================================
    # MENTORSHIP-specific slots (for mentors and mentees)
    # ========================================================================
    MENTORSHIP_SLOTS = [
        SlotDefinition(
            name="mentorship_areas",
            display_name="Mentorship Areas",
            slot_type=SlotType.MULTI_SELECT,
            description="What areas do you want guidance on?",
            required=True,
            depends_on=["primary_goal"],
            options=[
                "Leadership & Management",
                "Technical/Engineering",
                "Go-to-Market Strategy",
                "Fundraising",
                "Hiring & Team Building",
                "Product Development",
                "Sales & Business Development",
                "Marketing & Growth",
                "Operations & Scaling",
                "Career Development"
            ],
            extract_keywords=["leadership", "technical", "gtm", "fundraising", "hiring", "product", "sales", "marketing", "growth", "career", "management"]
        ),
        SlotDefinition(
            name="mentorship_format",
            display_name="Mentorship Format",
            slot_type=SlotType.SINGLE_SELECT,
            description="How would you like to connect?",
            required=False,
            depends_on=["primary_goal"],
            options=[
                "Weekly calls",
                "Bi-weekly calls",
                "Monthly sessions",
                "Async messaging",
                "Ad-hoc as needed"
            ],
            extract_keywords=["weekly", "monthly", "bi-weekly", "async", "calls", "messages", "ad-hoc", "flexible"]
        ),
        SlotDefinition(
            name="mentorship_commitment",
            display_name="Time Commitment",
            slot_type=SlotType.SINGLE_SELECT,
            description="Hours per month available",
            required=False,
            depends_on=["primary_goal"],
            options=[
                "1-2 hours/month",
                "3-5 hours/month",
                "5-10 hours/month",
                "10+ hours/month",
                "Flexible"
            ],
            extract_keywords=["hours", "month", "time", "commitment", "available", "availability"]
        ),
    ]

    # ========================================================================
    # COFOUNDER-specific slots (for finding co-founders)
    # ========================================================================
    COFOUNDER_SLOTS = [
        SlotDefinition(
            name="skills_have",
            display_name="Your Skills",
            slot_type=SlotType.MULTI_SELECT,
            description="What skills do you bring to a partnership?",
            required=True,
            depends_on=["primary_goal"],
            options=[
                "Technical/Engineering",
                "Product Management",
                "Sales/Business Development",
                "Marketing/Growth",
                "Finance/Operations",
                "Design/UX",
                "Domain Expertise",
                "Fundraising Experience"
            ],
            extract_keywords=["technical", "engineering", "product", "sales", "marketing", "finance", "design", "domain", "build", "code", "sell"]
        ),
        SlotDefinition(
            name="skills_need",
            display_name="Skills You Need",
            slot_type=SlotType.MULTI_SELECT,
            description="What complementary skills are you looking for?",
            required=True,
            depends_on=["primary_goal"],
            options=[
                "Technical/Engineering",
                "Product Management",
                "Sales/Business Development",
                "Marketing/Growth",
                "Finance/Operations",
                "Design/UX",
                "Domain Expertise",
                "Fundraising Experience"
            ],
            extract_keywords=["need", "looking for", "seeking", "want", "require", "complement"]
        ),
        SlotDefinition(
            name="commitment_level",
            display_name="Commitment Level",
            slot_type=SlotType.SINGLE_SELECT,
            description="What commitment level are you expecting?",
            required=True,
            depends_on=["primary_goal"],
            options=[
                "Full-time immediately",
                "Full-time after funding",
                "Part-time initially",
                "Nights & weekends",
                "Flexible/discuss"
            ],
            extract_keywords=["full-time", "part-time", "nights", "weekends", "flexible", "all-in", "committed"]
        ),
        SlotDefinition(
            name="equity_expectations",
            display_name="Equity Expectations",
            slot_type=SlotType.SINGLE_SELECT,
            description="How do you think about equity split?",
            required=False,
            depends_on=["primary_goal"],
            options=[
                "Equal split (50/50)",
                "Majority for existing founder",
                "Based on contribution",
                "Open to discuss",
                "Vesting with cliff"
            ],
            extract_keywords=["equity", "split", "50/50", "vesting", "ownership", "share", "stake"]
        ),
    ]

    # Advisor/Consultant-specific slots (for those OFFERING advisory services)
    ADVISOR_SLOTS = [
        SlotDefinition(
            name="advisory_type",
            display_name="Advisory Type",
            slot_type=SlotType.MULTI_SELECT,
            description="What type of advisory do you offer?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Strategic Advisory",
                "Technical Advisory",
                "Operational Advisory",
                "Board Member",
                "Fractional Executive",
                "Industry Expert"
            ],
            extract_keywords=["strategic", "technical", "operational", "board", "fractional", "cto", "cfo", "cmo", "advisor"]
        ),
        SlotDefinition(
            name="expertise_domains",
            display_name="Expertise Areas",
            slot_type=SlotType.MULTI_SELECT,
            description="What are your areas of expertise?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Product & Strategy",
                "Growth & Marketing",
                "Sales & BD",
                "Engineering & Tech",
                "Finance & Fundraising",
                "Operations & HR",
                "Legal & Compliance",
                "Industry-Specific"
            ],
            extract_keywords=["product", "growth", "marketing", "sales", "engineering", "finance", "fundraising", "operations", "legal", "compliance"]
        ),
        SlotDefinition(
            name="preferred_company_stage",
            display_name="Preferred Company Stages",
            slot_type=SlotType.MULTI_SELECT,
            description="What company stages do you prefer to advise?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Pre-seed", "Seed", "Series A", "Series B", "Series C+", "Growth", "Any Stage"
            ],
            extract_keywords=["pre-seed", "seed", "series", "growth", "early", "late", "stage"]
        ),
        SlotDefinition(
            name="compensation_model",
            display_name="Compensation Model",
            slot_type=SlotType.SINGLE_SELECT,
            description="How do you prefer to be compensated?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Equity only",
                "Cash only",
                "Hybrid (Cash + Equity)",
                "Pro-bono / Give-back",
                "Negotiable"
            ],
            extract_keywords=["equity", "cash", "fee", "retainer", "pro-bono", "free", "paid", "compensation"]
        ),
        SlotDefinition(
            name="time_commitment",
            display_name="Time Commitment",
            slot_type=SlotType.SINGLE_SELECT,
            description="How much time can you commit per month?",
            required=True,
            depends_on=["user_type"],
            options=[
                "1-5 hours/month",
                "5-10 hours/month",
                "10-20 hours/month",
                "20-40 hours/month",
                "40+ hours/month (Fractional)"
            ],
            extract_keywords=["hours", "time", "commitment", "availability", "fractional", "part-time"]
        ),
        SlotDefinition(
            name="advisory_track_record",
            display_name="Advisory Track Record",
            slot_type=SlotType.FREE_TEXT,
            description="How many companies have you advised?",
            required=False,
            depends_on=["user_type"],
            extract_keywords=["advised", "companies", "startups", "portfolio", "experience", "track record"]
        ),
    ]

    # Job Seeker/Candidate-specific slots (mirrors HIRING_SLOTS for matching)
    JOB_SEEKER_SLOTS = [
        SlotDefinition(
            name="target_role",
            display_name="Target Role",
            slot_type=SlotType.MULTI_SELECT,
            description="What roles are you looking for?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Engineering",
                "Product Management",
                "Design/UX",
                "Marketing",
                "Sales",
                "Operations",
                "Finance",
                "HR/People",
                "Legal",
                "Executive/C-Suite"
            ],
            extract_keywords=["engineer", "product", "design", "marketing", "sales", "operations", "finance", "hr", "legal", "executive", "role"]
        ),
        SlotDefinition(
            name="desired_seniority",
            display_name="Desired Seniority",
            slot_type=SlotType.SINGLE_SELECT,
            description="What seniority level are you targeting?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Entry Level",
                "Mid Level",
                "Senior",
                "Lead",
                "Manager",
                "Director",
                "VP",
                "C-Level"
            ],
            extract_keywords=["entry", "junior", "mid", "senior", "lead", "manager", "director", "vp", "chief", "executive"]
        ),
        SlotDefinition(
            name="salary_expectation",
            display_name="Salary Expectation",
            slot_type=SlotType.SINGLE_SELECT,
            description="What's your expected compensation range?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Under £50K",
                "£50K - £80K",
                "£80K - £120K",
                "£120K - £180K",
                "£180K - £250K",
                "£250K+",
                "Equity-focused",
                "Negotiable"
            ],
            extract_keywords=["salary", "compensation", "pay", "package", "equity", "stock", "options"]
        ),
        SlotDefinition(
            name="work_preference",
            display_name="Work Preference",
            slot_type=SlotType.SINGLE_SELECT,
            description="What's your preferred work arrangement?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Remote only",
                "Hybrid",
                "On-site",
                "Flexible"
            ],
            extract_keywords=["remote", "hybrid", "onsite", "office", "wfh", "flexible", "location"]
        ),
        SlotDefinition(
            name="availability",
            display_name="Availability",
            slot_type=SlotType.SINGLE_SELECT,
            description="When can you start a new role?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Immediately",
                "2 weeks notice",
                "1 month notice",
                "3 months notice",
                "Passive - open to opportunities"
            ],
            extract_keywords=["immediately", "available", "notice", "start", "when", "ready"]
        ),
        SlotDefinition(
            name="company_size_preference",
            display_name="Company Size Preference",
            slot_type=SlotType.MULTI_SELECT,
            description="What company sizes interest you?",
            required=False,
            depends_on=["user_type"],
            options=[
                "Early startup (<20)",
                "Startup (20-50)",
                "Scaleup (50-200)",
                "Mid-size (200-1000)",
                "Enterprise (1000+)",
                "Any size"
            ],
            extract_keywords=["startup", "scaleup", "enterprise", "small", "large", "size", "team"]
        ),
    ]

    # Partnership-specific slots (for Business Partnership objective)
    PARTNERSHIP_SLOTS = [
        SlotDefinition(
            name="partnership_type",
            display_name="Partnership Type",
            slot_type=SlotType.MULTI_SELECT,
            description="What type of partnership are you seeking?",
            required=True,
            depends_on=["primary_goal"],
            options=[
                "Distribution Partnership",
                "Technology Partnership",
                "Strategic Alliance",
                "Joint Venture",
                "Reseller/Channel Partner",
                "Co-marketing",
                "Integration Partner"
            ],
            extract_keywords=["distribution", "technology", "strategic", "alliance", "joint", "venture", "reseller", "channel", "integration"]
        ),
        SlotDefinition(
            name="resource_contribution",
            display_name="What You Bring",
            slot_type=SlotType.MULTI_SELECT,
            description="What resources can you contribute to a partnership?",
            required=True,
            depends_on=["primary_goal"],
            options=[
                "Capital/Investment",
                "Distribution/Channels",
                "Technology/Product",
                "Network/Relationships",
                "Brand/Marketing",
                "Operational Expertise",
                "Industry Knowledge"
            ],
            extract_keywords=["capital", "distribution", "technology", "network", "brand", "expertise", "bring", "contribute", "offer"]
        ),
        SlotDefinition(
            name="partnership_duration",
            display_name="Partnership Duration",
            slot_type=SlotType.SINGLE_SELECT,
            description="What partnership duration are you considering?",
            required=False,
            depends_on=["primary_goal"],
            options=[
                "Project-based (short-term)",
                "1 year",
                "Multi-year",
                "Open-ended/Ongoing",
                "Flexible"
            ],
            extract_keywords=["project", "year", "ongoing", "long-term", "short-term", "duration"]
        ),
        SlotDefinition(
            name="partner_size_preference",
            display_name="Ideal Partner Size",
            slot_type=SlotType.MULTI_SELECT,
            description="What size companies are you looking to partner with?",
            required=False,
            depends_on=["primary_goal"],
            options=[
                "Startups",
                "Scaleups",
                "Mid-market",
                "Enterprise",
                "Any size"
            ],
            extract_keywords=["startup", "scaleup", "mid-market", "enterprise", "size", "company"]
        ),
    ]

    # Service Provider-specific slots (for consultants, agencies, professionals)
    SERVICE_PROVIDER_SLOTS = [
        SlotDefinition(
            name="service_type",
            display_name="Service Type",
            slot_type=SlotType.MULTI_SELECT,
            description="What type of services do you offer?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Consulting",
                "Training/Coaching",
                "Implementation",
                "Managed Services",
                "Outsourcing",
                "Agency Services",
                "Speaking/Thought Leadership"
            ],
            extract_keywords=["consulting", "training", "coaching", "implementation", "managed", "outsourcing", "agency", "speaking"]
        ),
        SlotDefinition(
            name="pricing_model",
            display_name="Pricing Model",
            slot_type=SlotType.SINGLE_SELECT,
            description="How do you typically price your services?",
            required=True,
            depends_on=["user_type"],
            options=[
                "Hourly rate",
                "Project-based",
                "Retainer",
                "Revenue share",
                "Value-based",
                "Custom/Negotiable"
            ],
            extract_keywords=["hourly", "project", "retainer", "revenue", "share", "value", "pricing", "rate", "fee"]
        ),
        SlotDefinition(
            name="engagement_duration",
            display_name="Typical Engagement Duration",
            slot_type=SlotType.SINGLE_SELECT,
            description="How long are your typical engagements?",
            required=False,
            depends_on=["user_type"],
            options=[
                "Short-term (<1 month)",
                "Medium (1-6 months)",
                "Long-term (6+ months)",
                "Ongoing retainer",
                "Varies by project"
            ],
            extract_keywords=["short", "medium", "long", "ongoing", "retainer", "duration", "engagement"]
        ),
        SlotDefinition(
            name="client_size_preference",
            display_name="Ideal Client Size",
            slot_type=SlotType.MULTI_SELECT,
            description="What size clients do you prefer working with?",
            required=False,
            depends_on=["user_type"],
            options=[
                "Startups",
                "SMBs",
                "Mid-market",
                "Enterprise",
                "Any size"
            ],
            extract_keywords=["startup", "smb", "mid-market", "enterprise", "client", "size"]
        ),
    ]

    @classmethod
    def get_all_slots(cls) -> List[SlotDefinition]:
        """Get all slot definitions."""
        return (
            cls.CORE_SLOTS +
            cls.INVESTOR_SLOTS +
            cls.FOUNDER_SLOTS +
            cls.HIRING_SLOTS +
            cls.MENTORSHIP_SLOTS +
            cls.COFOUNDER_SLOTS +
            cls.ADVISOR_SLOTS +
            cls.JOB_SEEKER_SLOTS +
            cls.PARTNERSHIP_SLOTS +
            cls.SERVICE_PROVIDER_SLOTS +
            cls.OPTIONAL_SLOTS
        )

    @classmethod
    def get_slots_for_user_type(cls, user_type: str) -> List[SlotDefinition]:
        """Get relevant slots based on user type (legacy method)."""
        slots = list(cls.CORE_SLOTS)
        user_type_lower = user_type.lower() if user_type else ""

        if "investor" in user_type_lower:
            slots.extend(cls.INVESTOR_SLOTS)
        elif "founder" in user_type_lower or "entrepreneur" in user_type_lower:
            slots.extend(cls.FOUNDER_SLOTS)
        elif "recruiter" in user_type_lower or "hiring" in user_type_lower:
            slots.extend(cls.HIRING_SLOTS)
        elif "advisor" in user_type_lower or "consultant" in user_type_lower:
            slots.extend(cls.ADVISOR_SLOTS)
        elif "job seeker" in user_type_lower or "candidate" in user_type_lower:
            slots.extend(cls.JOB_SEEKER_SLOTS)
        elif "industry professional" in user_type_lower:
            slots.extend(cls.SERVICE_PROVIDER_SLOTS)

        slots.extend(cls.OPTIONAL_SLOTS)
        return slots

    @classmethod
    def get_slots_for_objective(cls, objective: str, user_type: str = None) -> List[SlotDefinition]:
        """
        Get slots dynamically based on primary_goal objective.

        This is the main method for objective-aware slot selection.
        Uses focus slots from use_case_templates.py when available.

        Args:
            objective: User's primary_goal value
            user_type: Optional user_type for additional filtering

        Returns:
            List of SlotDefinitions relevant to this objective
        """
        from app.services.use_case_templates import get_onboarding_slots

        objective_lower = objective.lower() if objective else ""

        # Get focus slots from use_case_templates
        try:
            focus_slot_names = get_onboarding_slots(objective_lower)
        except Exception:
            focus_slot_names = []

        # Universal slots always included (requirements/offerings are CRITICAL for matching)
        universal_slots = ["primary_goal", "user_type", "industry_focus", "geography", "dealbreakers", "requirements", "offerings"]

        # Start with universal slots from CORE_SLOTS
        slots = [s for s in cls.CORE_SLOTS if s.name in universal_slots]
        seen_names = {s.name for s in slots}

        # Map objectives to slot groups
        objective_slot_mapping = {
            # Investment flow
            "seeking investment": cls.FOUNDER_SLOTS,
            "fundraising": cls.FOUNDER_SLOTS,
            "looking to invest": cls.INVESTOR_SLOTS,
            "investing": cls.INVESTOR_SLOTS,
            # Hiring flow - EMPLOYER side
            "hiring talent": cls.HIRING_SLOTS,
            "hiring": cls.HIRING_SLOTS,
            # Mentorship flow - MENTEE side (seeking help)
            "seeking advisor/mentor": cls.MENTORSHIP_SLOTS,
            "mentorship": cls.MENTORSHIP_SLOTS,
            # Advisory flow - ADVISOR side (offering help)
            "offering advisory services": cls.ADVISOR_SLOTS,
            "advisory": cls.ADVISOR_SLOTS,
            # Cofounder flow
            "finding co-founder": cls.COFOUNDER_SLOTS,
            "cofounder": cls.COFOUNDER_SLOTS,
            # Partnership flow
            "business partnership": cls.PARTNERSHIP_SLOTS,
            "partnership": cls.PARTNERSHIP_SLOTS,
            # Networking is minimal (may get slots from user_type routing below)
            "networking": [],
        }

        # Add objective-specific slots
        for key, slot_group in objective_slot_mapping.items():
            if key in objective_lower:
                for slot in slot_group:
                    if slot.name not in seen_names:
                        slots.append(slot)
                        seen_names.add(slot.name)
                break

        # Add focus slots from templates that we have definitions for
        all_slots_by_name = {s.name: s for s in cls.get_all_slots()}
        for slot_name in focus_slot_names:
            if slot_name not in seen_names and slot_name in all_slots_by_name:
                slots.append(all_slots_by_name[slot_name])
                seen_names.add(slot_name)

        # Add relevant optional slots based on objective
        # BUG-039 FIX: Added recruiter, talent, service provider to get engagement_style
        optional_for_objective = {
            "seeking investment": ["engagement_style", "experience_years"],
            "looking to invest": ["engagement_style"],
            "mentorship": ["experience_years", "engagement_style"],
            "hiring": ["engagement_style"],
            "cofounder": ["engagement_style", "experience_years"],
            "partnership": ["engagement_style"],
            "networking": ["experience_years"],
            # BUG-039 FIX: Recruiters and service providers need engagement_style
            "recruiting": ["engagement_style", "experience_years"],
            "recruiter": ["engagement_style", "experience_years"],
            "talent": ["engagement_style", "experience_years"],
            "staffing": ["engagement_style", "experience_years"],
            "consulting": ["engagement_style", "experience_years"],
            "agency": ["engagement_style", "experience_years"],
            "service": ["engagement_style", "experience_years"],
        }

        for key, optional_names in optional_for_objective.items():
            if key in objective_lower:
                for opt_name in optional_names:
                    if opt_name not in seen_names:
                        opt_slot = cls.get_slot_by_name(opt_name)
                        if opt_slot:
                            slots.append(opt_slot)
                            seen_names.add(opt_name)
                break

        # Secondary routing: user_type can override or supplement objective-based slots
        # This handles cases where user_type is more specific than objective
        user_type_lower = (user_type or "").lower()

        # Job Seeker/Candidate always gets JOB_SEEKER_SLOTS regardless of objective
        # (they might select "Hiring Talent" meaning they want to BE hired)
        if "job seeker" in user_type_lower or "candidate" in user_type_lower:
            for slot in cls.JOB_SEEKER_SLOTS:
                if slot.name not in seen_names:
                    slots.append(slot)
                    seen_names.add(slot.name)

        # Industry Professional with networking/partnership objectives gets SERVICE_PROVIDER_SLOTS
        if "industry professional" in user_type_lower:
            if "networking" in objective_lower or not objective_slot_mapping.get(objective_lower):
                for slot in cls.SERVICE_PROVIDER_SLOTS:
                    if slot.name not in seen_names:
                        slots.append(slot)
                        seen_names.add(slot.name)

        return slots

    @classmethod
    def get_slot_by_name(cls, name: str) -> Optional[SlotDefinition]:
        """Get a slot definition by name."""
        for slot in cls.get_all_slots():
            if slot.name == name:
                return slot
        return None


class SlotExtractor:
    """
    Extracts slot values from natural language text.

    Uses pattern matching and keyword detection for extraction,
    with optional LLM enhancement for complex cases.
    """

    def __init__(self):
        self.schema = SlotSchema()

    def extract_from_text(
        self,
        text: str,
        target_slots: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ExtractedSlot]:
        """
        Extract slot values from text.

        Uses two-pass extraction to handle dependencies:
        - Pass 1: Extract slots without dependencies (e.g., user_type)
        - Pass 2: Extract dependent slots using newly extracted values as context

        Args:
            text: User's response text
            target_slots: Specific slots to extract (None = all applicable)
            context: Previous conversation context

        Returns:
            Dictionary of slot_name -> ExtractedSlot
        """
        text_lower = text.lower()
        results = {}

        # Determine which slots to extract
        all_slots = self.schema.get_all_slots()
        if target_slots:
            slots_to_extract = [s for s in all_slots if s.name in target_slots]
        else:
            slots_to_extract = all_slots

        # Build working context (merge existing context with new extractions)
        working_context = dict(context) if context else {}

        # Separate slots into those with and without dependencies
        independent_slots = [s for s in slots_to_extract if not s.depends_on]
        dependent_slots = [s for s in slots_to_extract if s.depends_on]

        # Pass 1: Extract independent slots first
        for slot_def in independent_slots:
            extracted = self._extract_slot(text, text_lower, slot_def)
            if extracted:
                results[slot_def.name] = extracted
                # Update working context so dependent slots can use this value
                working_context[slot_def.name] = extracted.value

        # Pass 2: Extract dependent slots using updated context
        for slot_def in dependent_slots:
            # Check if dependencies are now met (from context OR from Pass 1 results)
            dependencies_met = all(
                working_context.get(dep) is not None
                for dep in slot_def.depends_on
            )
            if not dependencies_met:
                continue

            # Extract based on slot type
            extracted = self._extract_slot(text, text_lower, slot_def)
            if extracted:
                results[slot_def.name] = extracted
                working_context[slot_def.name] = extracted.value

        return results

    def _extract_slot(
        self,
        text: str,
        text_lower: str,
        slot_def: SlotDefinition
    ) -> Optional[ExtractedSlot]:
        """Extract a single slot value."""

        if slot_def.slot_type == SlotType.SINGLE_SELECT:
            return self._extract_single_select(text, text_lower, slot_def)
        elif slot_def.slot_type == SlotType.MULTI_SELECT:
            return self._extract_multi_select(text, text_lower, slot_def)
        elif slot_def.slot_type == SlotType.NUMBER:
            return self._extract_number(text, text_lower, slot_def)
        elif slot_def.slot_type == SlotType.RANGE:
            return self._extract_range(text, text_lower, slot_def)
        elif slot_def.slot_type == SlotType.FREE_TEXT:
            return self._extract_free_text(text, text_lower, slot_def)
        elif slot_def.slot_type == SlotType.BOOLEAN:
            return self._extract_boolean(text, text_lower, slot_def)

        return None

    def _extract_single_select(
        self,
        text: str,
        text_lower: str,
        slot_def: SlotDefinition
    ) -> Optional[ExtractedSlot]:
        """Extract single selection from options."""
        best_match = None
        best_confidence = 0.0
        alternatives = []
        high_confidence_matches = []  # Track multiple strong matches (dual-role detection)

        # Fix for "Angel Investor" misclassification:
        # Detect if user is SEEKING investors (founder) vs BEING an investor

        # Strong founder indicators - these override investor keyword matches
        founder_indicators = [
            r"\b(ceo|founder|co-founder|cofounder)\b",
            r"\bi('m| am) (the )?(ceo|founder)",
            r"\bmy (company|startup|business)\b",
            r"\bwe('re| are) (raising|seeking|looking for)\b",
            r"\braise(d|ing)?\s+(a\s+)?\$?\d",
            r"\bseries [a-d]\b",
            r"\bseed (round|funding|stage)\b",
            r"\bfounded\b",
            r"\bour (team|company|startup)\b",
        ]
        is_founder = any(re.search(pattern, text_lower) for pattern in founder_indicators)

        # Phrases indicating user is SEEKING investors (they're a founder, not an investor)
        seeking_investor_patterns = [
            r"looking for\s+(\w+\s+)*investors?",  # "looking for US investors"
            r"seeking\s+(\w+\s+)*investors?",      # "seeking experienced investors"
            r"need\s+(\w+\s+)*investors?",
            r"find(ing)?\s+(\w+\s+)*investors?",
            r"attract(ing)?\s+(\w+\s+)*investors?",
            r"(raise|raising)\s+(funding|capital|money)",
            r"seeking\s+(investment|funding)",
            r"looking for\s+(funding|investment)",
            r"need\s+(funding|investment)",
            r"(want|wanting)\s+to\s+raise",
            r"fundrais(e|ing)",
        ]
        is_seeking_investor = any(re.search(pattern, text_lower) for pattern in seeking_investor_patterns)

        # Explicit correction detection: "I'm not an investor", "I am not an investor"
        investor_negation_patterns = [
            r"i('m| am) not (an? )?investor",
            r"not (an? )?investor",
            r"i('m| am) (a )?(founder|entrepreneur|ceo)",  # "I am a founder" is implicit correction
        ]
        is_explicitly_not_investor = any(re.search(pattern, text_lower) for pattern in investor_negation_patterns)

        # User is a founder if they show founder indicators OR are seeking investors OR explicitly said not investor
        user_is_founder = is_founder or is_seeking_investor or is_explicitly_not_investor

        for option in slot_def.options:
            option_lower = option.lower()

            # Skip investor options if user is clearly a founder
            if user_is_founder and slot_def.name == "user_type":
                if "investor" in option_lower:
                    continue  # Don't match Angel Investor, VC Partner, etc.

            # Exact match
            if option_lower in text_lower:
                confidence = 0.95
            # Partial word match - split on "/" and whitespace for options like "Founder/Entrepreneur"
            else:
                # Split option into individual words (handles "Founder/Entrepreneur", "VC Partner", etc.)
                option_words = [w for w in re.split(r'[/\s]+', option_lower) if len(w) > 2]
                # Use word boundaries to match each word
                word_matches = [w for w in option_words if re.search(r'\b' + re.escape(w) + r'\b', text_lower)]
                if word_matches:
                    confidence = 0.75 if len(word_matches) > 1 else 0.7
                # Keyword match
                elif any(kw in text_lower for kw in slot_def.extract_keywords):
                    # Check which option the keywords relate to
                    confidence = self._keyword_option_match(text_lower, option, slot_def.extract_keywords)
                else:
                    continue

            if confidence > 0:
                alternatives.append((option, confidence))
                # Track high-confidence matches for dual-role detection
                if confidence >= 0.65:
                    high_confidence_matches.append((option, confidence))
                if confidence > best_confidence:
                    best_match = option
                    best_confidence = confidence

        if best_match:
            # Bug 1 Fix: Detect dual-role ambiguity (e.g., "I'm both an investor and founder")
            # If multiple high-confidence matches exist, lower confidence to trigger clarification
            if len(high_confidence_matches) > 1 and slot_def.name == "user_type":
                # Check for explicit dual-role indicators
                dual_role_indicators = ["both", "and also", "as well as", "plus"]
                has_dual_role_indicator = any(ind in text_lower for ind in dual_role_indicators)

                if has_dual_role_indicator:
                    # User explicitly stated multiple roles - set low confidence to ask for primary
                    best_confidence = 0.45
                    logger.info(f"Dual-role detected for user_type: {[m[0] for m in high_confidence_matches]}")

            return ExtractedSlot(
                name=slot_def.name,
                value=best_match,
                confidence=best_confidence,
                status=SlotStatus.FILLED if best_confidence > 0.7 else SlotStatus.PARTIAL,
                source_text=text,
                alternatives=[(a, c) for a, c in alternatives if a != best_match][:3]
            )

        return None

    def _extract_multi_select(
        self,
        text: str,
        text_lower: str,
        slot_def: SlotDefinition
    ) -> Optional[ExtractedSlot]:
        """Extract multiple selections from options."""
        matches = []
        total_confidence = 0.0

        # Bug 3 Fix: Track which text positions are already matched to avoid overlapping matches
        # For example, "fintech" should not also match "tech" from "Clean Tech"
        matched_positions = set()

        for option in slot_def.options:
            option_lower = option.lower()

            # Check for exact option match (e.g., "fintech" matches "Fintech")
            # Use word boundary matching to avoid substring issues
            exact_pattern = r'\b' + re.escape(option_lower) + r'\b'
            exact_match = re.search(exact_pattern, text_lower)

            if exact_match:
                # Check if this position overlaps with already matched text
                match_start, match_end = exact_match.span()
                if not any(pos in matched_positions for pos in range(match_start, match_end)):
                    matches.append(option)
                    total_confidence += 0.9
                    # Mark these positions as matched
                    matched_positions.update(range(match_start, match_end))
                continue

            # For partial word matches, require the word to be standalone AND significant
            # Skip common short words like "tech", "ai" that cause false positives
            option_words = option_lower.split()
            matched_word = False

            for word in option_words:
                # Skip words that are too common/short and cause false positives
                if len(word) <= 4 and word in ['tech', 'ai', 'ml', 'b2b', 'b2c']:
                    # These short words should only match if they appear as standalone
                    word_pattern = r'\b' + re.escape(word) + r'\b'
                    if not re.search(word_pattern, text_lower):
                        continue
                    # Found as standalone word - but check it's not part of a compound word already matched
                    word_match = re.search(word_pattern, text_lower)
                    if word_match and any(pos in matched_positions for pos in range(word_match.start(), word_match.end())):
                        continue

                if len(word) > 4:
                    # For longer words, check if they appear as a word boundary match
                    word_pattern = r'\b' + re.escape(word) + r'\b'
                    word_match = re.search(word_pattern, text_lower)
                    if word_match:
                        # Check not overlapping with existing matches
                        if not any(pos in matched_positions for pos in range(word_match.start(), word_match.end())):
                            matched_word = True
                            break

            if matched_word:
                matches.append(option)
                total_confidence += 0.7

        if matches:
            avg_confidence = total_confidence / len(matches)
            return ExtractedSlot(
                name=slot_def.name,
                value=matches,
                confidence=min(0.95, avg_confidence),
                status=SlotStatus.FILLED if avg_confidence > 0.6 else SlotStatus.PARTIAL,
                source_text=text
            )

        return None

    def _extract_number(
        self,
        text: str,
        text_lower: str,
        slot_def: SlotDefinition
    ) -> Optional[ExtractedSlot]:
        """Extract numeric value."""
        # Look for numbers with optional k/m/b suffixes
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:million|m\b)',
            r'(\d+(?:\.\d+)?)\s*(?:thousand|k\b)',
            r'(\d+(?:\.\d+)?)\s*(?:billion|b\b)',
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'£\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:years?|employees?|people)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                value_str = match.group(1).replace(',', '')
                value = float(value_str)

                # Apply multiplier
                if 'million' in text_lower or 'm' in pattern:
                    value *= 1_000_000
                elif 'thousand' in text_lower or 'k' in pattern:
                    value *= 1_000
                elif 'billion' in text_lower or 'b' in pattern:
                    value *= 1_000_000_000

                # Validate against min/max
                if slot_def.min_value is not None and value < slot_def.min_value:
                    continue
                if slot_def.max_value is not None and value > slot_def.max_value:
                    continue

                return ExtractedSlot(
                    name=slot_def.name,
                    value=value,
                    confidence=0.85,
                    status=SlotStatus.FILLED,
                    source_text=text
                )

        return None

    def _extract_range(
        self,
        text: str,
        text_lower: str,
        slot_def: SlotDefinition
    ) -> Optional[ExtractedSlot]:
        """Extract numeric range."""
        # Look for range patterns (with units)
        range_patterns_with_units = [
            r'(\d+(?:\.\d+)?)\s*(?:k|m|million|thousand)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:k|m|million|thousand)',
            r'between\s*(?:\$|£)?\s*(\d+(?:\.\d+)?)\s*(?:k|m)\s*and\s*(?:\$|£)?\s*(\d+(?:\.\d+)?)\s*(?:k|m)',
            # Handle "$5-10M" format (unit only on second number)
            r'(?:\$|£)\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:m|k|million|thousand)',
            # Handle "5-10 million" format (unit as separate word after)
            r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s+(?:million|thousand|m|k)\b',
        ]

        for pattern in range_patterns_with_units:
            match = re.search(pattern, text_lower)
            if match:
                min_val = float(match.group(1).replace(',', ''))
                max_val = float(match.group(2).replace(',', ''))

                # Apply multipliers based on context
                if 'm' in text_lower or 'million' in text_lower:
                    if min_val < 1000:
                        min_val *= 1_000_000
                    if max_val < 1000:
                        max_val *= 1_000_000
                elif 'k' in text_lower or 'thousand' in text_lower:
                    if min_val < 10000:
                        min_val *= 1_000
                    if max_val < 10000:
                        max_val *= 1_000

                return ExtractedSlot(
                    name=slot_def.name,
                    value={"min": min_val, "max": max_val},
                    confidence=0.85,
                    status=SlotStatus.FILLED,
                    source_text=text
                )

        # Bug 2 Fix: Handle ranges without units (e.g., "between 1 and 5")
        # These get low confidence since units are ambiguous
        naked_range_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\b',
            r'between\s*(?:\$|£)?\s*(\d+(?:\.\d+)?)\s*and\s*(?:\$|£)?\s*(\d+(?:\.\d+)?)\b',
        ]

        for pattern in naked_range_patterns:
            match = re.search(pattern, text_lower)
            if match:
                min_val = float(match.group(1).replace(',', ''))
                max_val = float(match.group(2).replace(',', ''))

                # Check if any unit indicators are present as standalone units (word boundaries)
                # This prevents false positives like "looking" matching "k"
                has_million = bool(re.search(r'\bm\b|\bmillion\b', text_lower))
                has_thousand = bool(re.search(r'\bk\b|\bthousand\b', text_lower))

                if has_million:
                    if min_val < 1000:
                        min_val *= 1_000_000
                    if max_val < 1000:
                        max_val *= 1_000_000
                    confidence = 0.85
                elif has_thousand:
                    if min_val < 10000:
                        min_val *= 1_000
                    if max_val < 10000:
                        max_val *= 1_000
                    confidence = 0.85
                else:
                    # No units specified - extract with low confidence to trigger clarification
                    # Keep raw values, system should ask for clarification
                    confidence = 0.35
                    logger.info(f"Range without units detected: {min_val} to {max_val} - needs clarification")

                return ExtractedSlot(
                    name=slot_def.name,
                    value={"min": min_val, "max": max_val},
                    confidence=confidence,
                    status=SlotStatus.FILLED if confidence > 0.7 else SlotStatus.PARTIAL,
                    source_text=text
                )

        # Try to extract single number as a point estimate
        number_result = self._extract_number(text, text_lower, slot_def)
        if number_result:
            # Convert to range with ±20%
            val = number_result.value
            number_result.value = {"min": val * 0.8, "max": val * 1.2}
            number_result.confidence *= 0.8  # Lower confidence for inferred range
            return number_result

        return None

    def _extract_free_text(
        self,
        text: str,
        text_lower: str,
        slot_def: SlotDefinition
    ) -> Optional[ExtractedSlot]:
        """Extract free-form text if relevant keywords present."""
        # Check if any extract keywords are present
        keyword_found = any(kw in text_lower for kw in slot_def.extract_keywords)

        if keyword_found and len(text) > 20:
            return ExtractedSlot(
                name=slot_def.name,
                value=text.strip(),
                confidence=0.7,
                status=SlotStatus.FILLED,
                source_text=text
            )

        return None

    def _extract_boolean(
        self,
        text: str,
        text_lower: str,
        slot_def: SlotDefinition
    ) -> Optional[ExtractedSlot]:
        """Extract yes/no value."""
        positive = ["yes", "yeah", "yep", "sure", "definitely", "absolutely", "correct", "true"]
        negative = ["no", "nope", "not", "never", "false", "don't", "won't"]

        for word in positive:
            if word in text_lower:
                return ExtractedSlot(
                    name=slot_def.name,
                    value=True,
                    confidence=0.9,
                    status=SlotStatus.FILLED,
                    source_text=text
                )

        for word in negative:
            if word in text_lower:
                return ExtractedSlot(
                    name=slot_def.name,
                    value=False,
                    confidence=0.9,
                    status=SlotStatus.FILLED,
                    source_text=text
                )

        return None

    def _keyword_option_match(
        self,
        text_lower: str,
        option: str,
        keywords: List[str]
    ) -> float:
        """Calculate confidence based on keyword-option correlation."""
        # Simple heuristic - could be enhanced with embeddings
        option_lower = option.lower()

        for keyword in keywords:
            if keyword in text_lower:
                # Check if keyword relates to this option
                if keyword in option_lower:
                    return 0.75
                # Common associations
                associations = {
                    "invest": ["investor", "investment", "looking to invest"],
                    "funding": ["seeking investment", "raise", "capital"],
                    "founder": ["founder", "entrepreneur", "startup"],
                    "mentor": ["advisor", "mentor", "advisory"],
                }
                for kw, opts in associations.items():
                    if keyword == kw and any(o in option_lower for o in opts):
                        return 0.65

        return 0.0


# Global instance
slot_extractor = SlotExtractor()
