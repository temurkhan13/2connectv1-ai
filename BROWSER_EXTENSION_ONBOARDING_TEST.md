# Claude Browser Extension - Onboarding Test Instructions

## Test User Credentials

| Field | Value |
|-------|-------|
| **Email** | `debug.1770850581@test.com` |
| **Password** | `Test123!@#` |
| **User ID** | `ef180c57-4885-4b75-a527-87df19e2a2a0` |
| **Name** | Debug User |

## Application URL

```
http://localhost:5173
```

## Step-by-Step Instructions

### Step 1: Navigate and Sign In

1. Go to `http://localhost:5173`
2. Click "Sign In" or navigate to the login page
3. Enter credentials:
   - Email: `debug.1770850581@test.com`
   - Password: `Test123!@#`
4. Click "Sign In" button

### Step 2: Start Onboarding

After signing in, you should be redirected to the onboarding flow. If not, navigate to `/onboarding`.

### Step 3: Complete Conversational Onboarding

The onboarding is a chat-based flow. Respond to the AI's questions using the persona below.

## Test Persona: Sarah Chen - Startup Founder

Use this information to answer the onboarding questions:

### Basic Information
- **Name**: Sarah Chen
- **Role**: Founder & CEO
- **Company**: TechFlow AI
- **Industry**: Artificial Intelligence / SaaS

### Professional Background
- Founded TechFlow AI 2 years ago
- Previously worked as Senior Product Manager at Google for 5 years
- MBA from Stanford, BS in Computer Science from MIT
- Based in San Francisco, CA

### Current Situation
- **User Type**: Founder/Entrepreneur seeking funding
- **Funding Stage**: Pre-seed, preparing for Seed round
- **Funding Need**: Looking to raise $2M seed round
- **Timeline**: Within the next 3-6 months

### What They're Looking For
- Angel investors with AI/ML expertise
- Seed-stage VCs focused on B2B SaaS
- Mentors with enterprise sales experience
- Technical advisors in machine learning

### What They Offer
- Deep AI/ML product expertise
- Enterprise go-to-market experience
- Strong technical team (5 engineers)
- Early traction: $50K MRR, 15 enterprise customers

### Goals & Interests
- Scale the engineering team
- Expand enterprise customer base
- Build strategic partnerships
- Prepare for Series A in 18 months

## Sample Responses for Chat

When the AI asks questions, respond naturally using the persona. Here are example responses:

**Q: Tell me about yourself**
> "Hi! I'm Sarah Chen, founder and CEO of TechFlow AI. We're building an AI-powered workflow automation platform for enterprise teams. I previously spent 5 years at Google as a Senior Product Manager before starting this company 2 years ago."

**Q: What brings you to Reciprocity?**
> "I'm looking to raise our seed round - targeting $2M in the next 3-6 months. I'd love to connect with investors who have deep expertise in AI/ML and B2B SaaS. Also interested in finding mentors who've scaled enterprise sales teams."

**Q: What stage is your company?**
> "We're at the pre-seed stage, preparing for our seed round. We have early traction with $50K MRR and 15 enterprise customers. Our team is 5 engineers plus myself."

**Q: What industry are you in?**
> "Artificial Intelligence and SaaS. Specifically, we're building workflow automation tools powered by large language models for enterprise productivity."

**Q: What are you looking for in connections?**
> "Primarily investors - both angels with AI expertise and seed-stage VCs focused on B2B SaaS. I'm also looking for mentors who have experience scaling enterprise sales, and technical advisors in machine learning."

**Q: What can you offer to others?**
> "I can share insights on building AI products, enterprise go-to-market strategies, and navigating the transition from big tech to startups. Happy to help other founders with product development and early customer acquisition."

## Expected Outcomes

After completing onboarding, verify:

1. **Slots Extracted** (should be 8/8):
   - `user_type`: founder
   - `name`: Sarah Chen
   - `industry`: AI/SaaS
   - `role`: Founder/CEO
   - `company`: TechFlow AI
   - `funding_stage`: pre-seed
   - `funding_need`: $2M seed
   - `timeline`: 3-6 months

2. **AI Summary Generated**: Check that the user appears in Discover page

3. **Role Detection**: Summary should identify user as "Founder" not "Investor"

## Verification Steps

After onboarding completes:

1. Navigate to **Dashboard** - should see personalized greeting
2. Navigate to **Discover** - user's profile should be searchable
3. Navigate to **Inbox** - should load without infinite spinner (may show empty state if no messages)
4. Navigate to **Settings** - should show user profile data

## Troubleshooting

If onboarding fails:
- Check browser console for errors
- Verify AI service is running: `curl http://localhost:8000/health`
- Verify backend is running: `curl http://localhost:3000`
- Check Docker containers: `docker compose ps`
