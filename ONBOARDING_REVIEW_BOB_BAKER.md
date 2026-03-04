# Onboarding Session Review: Bob Baker (Angel Investor)
**Date:** 2026-03-03
**Test Type:** Real user onboarding session analysis

---

## User Profile Summary

**Name:** Bob Baker
**Role:** Angel Investor
**Location:** New York City
**Investment Profile:**
- Portfolio: 40+ investments over 8 years
- Focus sectors: HealthTech, CleanTech, Deep Tech
- Check size: $100K-$500K
- Stage: Pre-seed to seed
- Track record: 6 exits (2 acquisitions, 4 IPOs), 34% IRR
- Available capital: $2M over next 12 months

**Investment Thesis:**
- "Picks and shovels" approach (infrastructure/tools, not applications)
- HealthTech: AI diagnostics, remote monitoring (cost reduction)
- CleanTech: Battery storage, grid optimization
- Deep Tech: Quantum computing applications, novel materials
- Criteria: Domain expertise, clear revenue path, execution capability

**Seeking:**
- Founders in target sectors
- Co-investors (angels/family offices)
- Advisory board opportunities

---

## Conversation Flow Analysis

| Turn | Speaker | Type | Content |
|------|---------|------|---------|
| 1 | System | Greeting + Initial Q | "Tell me about yourself - what you're working on and what kind of connections you're looking for" |
| 2 | Bob | Answer | Comprehensive intro (role, portfolio, sectors, check size, seeking) |
| 3 | System | Follow-up Q | "What are some of the key things you look for when evaluating..." |
| 4 | Bob | Answer | Investment thesis, criteria, track record, available capital |
| 5 | System | Conclusion | "Perfect! I have everything I need..." |

**Total Questions Asked:** 2

---

## Critical Issues Identified

### ❌ ISSUE 1: Severely Under-Extracted Profile

**Severity:** CRITICAL

**Problem:**
- Only 2 questions asked vs expected 5-7 questions
- System concluded prematurely without gathering critical matching criteria
- This is WORSE than previous test reports (Alice: 7 questions, Anonymous: 6 questions)

**Missing Critical Information:**

1. **Friction Points / Deal Breakers**
   - What sectors/approaches Bob specifically avoids
   - Red flags in team composition
   - Revenue requirements before investment
   - Minimum/maximum team size
   - Geographic restrictions (does he only invest in US?)

2. **Advisory Board Criteria**
   - Time commitment Bob can offer
   - What kind of value he brings (beyond capital)
   - Industries where he has operational experience
   - Current advisory commitments/capacity

3. **Co-Investor Preferences**
   - What makes a good co-investor for Bob
   - Investment committee dynamics
   - Due diligence approach
   - Post-investment involvement expectations

4. **Communication Preferences**
   - How Bob wants to be approached by founders
   - Pitch format preferences (deck, demo, intro call)
   - Response time expectations
   - Follow-up cadence

5. **Portfolio Strategy**
   - Current portfolio gaps he's looking to fill
   - Diversification goals
   - Follow-on investment strategy
   - Exit timeline preferences

6. **Founder Criteria**
   - Technical vs business co-founder balance
   - Prior exit experience importance
   - Full-time commitment requirements
   - Location flexibility

**Impact:**
- Matches will be lower quality without friction point data
- Risk of introducing Bob to founders he'd immediately reject
- Can't personalize match explanations without deeper criteria
- Advisory board matches unclear without time/value parameters

---

## Comparison to Previous Test Results

| Test | Questions Asked | Quality | Issues |
|------|----------------|---------|--------|
| Alice Anderson (Report 1) | 7 | Good | Onboarding loop bug, but extraction OK |
| Anonymous (Report 2) | 6 | Good | AI Summary blank, but extraction OK |
| **Bob Baker (This session)** | **2** | **Poor** | **Severe under-extraction** |

**Trend:** This session shows significantly worse extraction than previous tests, suggesting:
1. Extraction improvements (commit 9cb1f43) may not be consistently applied
2. System may have different behavior for different user types
3. Backend validation may have failed to detect incomplete profile

---

## Quality Assessment

### ✅ What Worked

1. **Good initial question** - Open-ended, allowed comprehensive first response
2. **Relevant follow-up** - Asked about investment thesis (good depth)
3. **User provided detail** - Bob gave rich responses with specifics

### ❌ What Failed

1. **Premature conclusion** - 2 questions is insufficient for quality matching
2. **No friction point extraction** - Critical for investor matching
3. **No validation questions** - System should have asked "What should I avoid showing you?"
4. **No capacity assessment** - Time/capital availability unclear
5. **No preference confirmation** - System didn't validate understanding

---

## Expected vs Actual Question Flow

### Expected Flow (5-7 questions):
1. ✅ Initial intro question
2. ✅ Investment thesis/criteria
3. ❌ Friction points/deal breakers *(MISSING)*
4. ❌ Advisory board criteria *(MISSING)*
5. ❌ Co-investor preferences *(MISSING)*
6. ❌ Communication/approach preferences *(MISSING)*
7. ❌ Validation/confirmation *(MISSING)*

### Actual Flow (2 questions):
1. ✅ Initial intro
2. ✅ Investment thesis
3. ❌ **CONCLUDED PREMATURELY**

---

## Recommended Investigation

1. **Check extraction service logs** for this session:
   ```bash
   # If session_id available, check extraction_results
   grep "session_id" reciprocity-ai/logs/*.log | grep "extraction"
   ```

2. **Verify extraction hints applied:**
   - Check if `ConversationalQuestionService` used enhanced hints (commit 9cb1f43)
   - Verify `question_limit` validation occurred
   - Check if backend profile validation passed

3. **Test reproduction:**
   - Create test user with investor profile
   - Run through onboarding
   - Count questions and check logs

4. **Backend profile validation:**
   - Check if `create_profile()` endpoint validated completeness
   - Verify minimum field requirements enforced

---

## Recommended Fixes

### Fix 1: Enforce Minimum Question Count
**Priority:** HIGH

**File:** `reciprocity-ai/app/services/conversational_question_service.py`

Add validation before conclusion:
```python
MIN_QUESTIONS = 5

def should_conclude_conversation(self, session_data: dict) -> bool:
    questions_asked = len(session_data.get('messages', [])) // 2  # User + AI pairs

    if questions_asked < MIN_QUESTIONS:
        logger.warning(f"Attempted conclusion with only {questions_asked} questions (min: {MIN_QUESTIONS})")
        return False  # Force more questions

    # Existing logic...
```

### Fix 2: Add Friction Point Detection
**Priority:** HIGH

Enhance extraction hints to explicitly ask for friction points:
```python
EXTRACTION_HINTS = """
CRITICAL: Always extract friction points and deal breakers.
For investors: sectors/stages they AVOID, red flags, geographic limits
For founders: investor types they DON'T want, terms they reject
For professionals: company types they'd never join

If user doesn't mention friction points, ASK EXPLICITLY:
"What should I definitely NOT show you?"
"""
```

### Fix 3: Add Profile Completeness Check
**Priority:** MEDIUM

**File:** `reciprocity-backend/src/modules/onboarding/onboarding.service.ts`

Before creating profile, validate completeness:
```typescript
validateProfileCompleteness(data: CreateProfileDto): ValidationResult {
  const required = ['goals', 'friction_points', 'preferences'];
  const missing = required.filter(field => !data[field] || data[field].length === 0);

  if (missing.length > 0) {
    return {
      valid: false,
      message: `Incomplete profile: missing ${missing.join(', ')}`,
      missing_fields: missing
    };
  }

  return { valid: true };
}
```

---

## Success Criteria for Fix Validation

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| Minimum questions | 5-7 questions | Count questions in transcript |
| Friction points captured | At least 3 deal breakers | Check `profile.friction_points` field |
| Preferences captured | At least 3 preferences | Check `profile.preferences` field |
| No premature conclusion | System asks follow-up if info missing | Attempt incomplete answers |
| Validation works | Backend rejects incomplete profiles | Check API response |

---

## Next Steps

1. **Immediate:** Check backend logs for this session to see extraction results
2. **Investigation:** Test with new investor profile to reproduce issue
3. **Fix:** Implement minimum question count + friction point detection
4. **Validation:** Run 3 test onboardings and verify 5-7 questions asked
5. **Deploy:** Push fixes to staging and re-test with real users

---

## Related Documents

- Previous user testing: `USER_TESTING_ANALYSIS_2026-03-03.md`
- Extraction improvements: `IMPLEMENTATION_SUMMARY.md` (commit 9cb1f43)
- Deployed fixes: `BUGFIXES_2026-03-03.md`
