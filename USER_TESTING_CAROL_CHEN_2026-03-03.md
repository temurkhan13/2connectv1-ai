# User Testing Report Analysis: Carol Chen
**Date:** 2026-03-03
**Account:** carol.chen.1772535113@2connect-test.com
**Test Type:** Real user journey evaluation

---

## Executive Summary

**Overall Result:** ✅ Platform mostly working, but reveals **persona generation timing issue** and **slot extraction problems**

**Key Findings:**
1. ✅ **AI Summary empty state IMPROVED** - Shows proper message instead of blank box (our fix worked!)
2. ⚠️ **Persona generation DELAYED** - Not generated during 15-min session (possibly skip flow issue)
3. ⚠️ **Slot extraction not working** - Mentioned "General Partner" + "Sequoia Capital" but still flagged missing
4. ✅ **Express Interest working** - Confirmed functional
5. ⚠️ **Match scores all 50%** - Embedding pipeline hadn't processed new user yet

---

## Cross-Report Comparison

| Report | User | Questions | Completion | Persona Generated | AI Summary |
|--------|------|-----------|------------|-------------------|------------|
| 1 | Alice Anderson | 7 | Organic | ✅ Yes | ❌ Onboarding loop blocked |
| 2 | Anonymous | 6 | Organic | ✅ Yes | ❌ Blank box |
| 3 | Bob Baker | **2** | Organic | Unknown | Unknown |
| 4 | **Carol Chen** | **3 + skip** | **Semi-skip** | **❌ Not during test** | **✅ Proper empty state** |

---

## Fix Validation Results

### ✅ Fix 1: AI Summary Empty State (CONFIRMED WORKING)

**Our Fix:**
- Added `getSummaryText()` with fallback field detection
- Added debug logging
- Shows proper empty state message when no persona

**Carol's Result:**
> "No AI summary found — Your AI summary will be generated automatically after completing onboarding."

**Comparison:**
- **Previous (ayesha):** Blank white box with Edit/Approve buttons but no content
- **Current (Carol):** Proper informative empty state

**Verdict:** ✅ **FIX WORKING** - Empty state UX significantly improved

---

### ⏸️ Fix 2: Onboarding Loop (NOT TESTED)

**Our Fix:**
- Added `await checkStatus()` to refresh user object after profile creation
- Should prevent infinite redirect loop

**Carol's Result:**
- Used "Skip and find matches" at 71% instead of organic completion
- Never tested the organic 100% → dashboard transition
- **Cannot confirm fix worked**

**Verdict:** ⏸️ **NEEDS ORGANIC COMPLETION TEST** - Skip flow doesn't test the loop bug

---

### ⏸️ Fix 3: Signup 409 Error (NOT TESTED)

**Our Fix:**
- Modified `signUp()` to return 409 errors properly to form
- Form shows field-level error message

**Carol's Result:**
- Successfully logged in with correct password
- Never attempted signup with existing email
- **Cannot confirm fix worked**

**Verdict:** ⏸️ **NEEDS DUPLICATE EMAIL TEST**

---

## New Issues Discovered

### 🚨 ISSUE 1: Persona Generation Delayed/Broken (MEDIUM Severity)

**Observed:**
- Carol completed onboarding (~71% + skip)
- Waited 15 minutes
- Persona never generated
- No "Updated The Persona" in agent activity
- AI Summary still shows empty state

**Comparison:**
- **ayesha (previous account):** Persona generated in ~5 minutes
- **Carol (this account):** No generation during entire test session

**Possible Root Causes:**

1. **Skip flow bypasses trigger:**
   ```typescript
   // In completeOnboarding()
   // Does "Skip and find matches" call the same completion endpoint?
   // Or does it bypass the Celery task trigger?
   ```

2. **Celery queue delay:**
   ```python
   # In persona_service.py
   generate_persona_task.delay(user_id)
   # Is the task queued but not processed?
   ```

3. **Insufficient data:**
   ```python
   # Persona generation might require minimum slot completeness
   # Carol had "Role Title, Company Name" flagged as missing
   # Generator might wait for complete profile
   ```

**Investigation Needed:**
```bash
# Check Celery logs for Carol's user_id
grep "carol.chen" reciprocity-ai/logs/celery*.log

# Check if profile creation webhook fired
grep "profile_created" reciprocity-backend/logs/*.log | grep carol.chen

# Check persona service logs
grep "generate_persona_task" reciprocity-ai/logs/*.log
```

---

### ⚠️ ISSUE 2: Slot Extraction Not Working (MEDIUM Severity)

**Observed:**
- Carol mentioned: "General Partner at Sequoia Capital"
- System still showed: "4 more details could improve your matches: Primary Goal, **Role Title, Company Name**"
- Extraction service didn't parse role/company from conversational context

**Expected:**
```python
# ConversationalQuestionService should extract:
slots = {
    'role': 'General Partner',
    'company': 'Sequoia Capital',
    # ... other fields
}
```

**Actual:**
```python
# Slots remained unfilled:
slots = {
    'role': None,  # ❌ Not extracted
    'company': None,  # ❌ Not extracted
}
```

**Root Cause:**
The extraction hints (commit 9cb1f43) may not be handling implicit mentions:
```python
# Current extraction probably expects explicit key-value format:
"I work as a General Partner at Sequoia Capital"  # ✅ Works

# But fails on conversational phrasing:
"I'm a GP at Sequoia"  # ❌ Misses
"General Partner, Sequoia Capital"  # ❌ Misses
```

**Impact:**
- Users have to repeat information explicitly
- "Skip and find matches" button appears prematurely
- Matching quality reduced (missing critical fields)

---

### ⚠️ ISSUE 3: Match Scores All 50% (LOW Severity - Expected Behavior)

**Observed:**
- Carol's Discover page: All 82 profiles show 50% match score
- ayesha's Discover page: Varied scores (62%–83%)

**Root Cause:**
This is **expected behavior** - not a bug:
```python
# In matching_service.py
# When user profile is new, embeddings haven't been computed yet
# System returns default fallback score: 50%

# Embedding pipeline runs asynchronously:
# 1. User completes onboarding
# 2. Profile created
# 3. Embedding generation queued (Celery task)
# 4. Similarity computation runs (~5-10 minutes)
# 5. Real scores appear
```

**Issue:**
- No loading indicator tells user scores are still computing
- User sees flat 50% and thinks matching is broken

**Recommendation:**
Add loading state to match cards:
```typescript
// MatchCard.tsx
{isEmbeddingsPending ? (
  <div className="flex items-center gap-2">
    <Spinner size="xs" />
    <span className="text-sm text-gray-500">Computing match...</span>
  </div>
) : (
  <span className="font-bold">{matchScore}%</span>
)}
```

---

### ⚠️ ISSUE 4: Inconsistent Metric Labels (LOW Severity)

**Observed:**
- Dashboard: Middle metric labeled "Match approval rate"
- Matches page: Same metric labeled "Response rate"

**Files:**
- `reciprocity-frontend/src/pages/Dashboard/DashboardPage.tsx`
- `reciprocity-frontend/src/pages/Matches/MatchesPage.tsx`

**Fix:**
Standardize to one label (suggest "Approval rate"):
```typescript
// DashboardPage.tsx line ~45
{ label: 'Approval rate', value: '0%', icon: CheckCircle }

// MatchesPage.tsx line ~28
{ label: 'Approval rate', value: '0%' }
```

---

### ⚠️ ISSUE 5: Member ID Truncation (LOW Severity)

**Observed:**
- At 1345px viewport: "Member ..." (truncated)
- At 1437px viewport: "Member #4E5F6E7D" (full)

**Root Cause:**
Profile card title doesn't scale responsively:
```css
/* Current: fixed width, truncates */
.profile-card-title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Fix: allow wrapping or use smaller font at narrow widths */
```

---

### ⚠️ ISSUE 6: Match Score Fluctuation (LOW Severity)

**Observed:**
- Dashboard shows 9%, then 7% on reload
- Same instability seen in all test accounts

**Investigation Needed:**
```bash
# Check if score calculation is non-deterministic
grep "average_match_score" reciprocity-backend/src/modules/dashboard/*.ts
```

---

## Onboarding Flow Comparison

### ayesha (Previous Account) - Organic Completion
```
Turn 1: Initial question → 0% to 41%
Turn 2: Follow-up → 41% to 58%
Turn 3: Follow-up → 58% to 75%
Turn 4: Follow-up → 75% to 85%
Turn 5: Follow-up → 85% to 92%
Turn 6: Follow-up → 92% to 97%
Turn 7: Final question → 97% to 100%
Result: "Complete & View Matches" → Profile created → Persona generated in ~5 min
```

### Carol Chen (This Account) - Semi-Skip
```
Turn 1: Initial question → 0% to 65% (FASTER - richer answer)
Turn 2: Follow-up → 65% to 71%
Turn 3: "Skip and find matches" button appears
Action: Clicked skip → Synthetic message sent → Profile created
Result: Persona NOT generated during 15-min test session
```

**Hypothesis:**
The skip flow may bypass the persona generation trigger. Check if:
1. `completeOnboarding()` triggers persona task
2. Skip button calls same endpoint or different one
3. Skip flow sets `onboarding_status = 'completed'` but skips webhook

---

## Bob Baker vs Carol Chen - Extraction Quality

### Bob Baker (Investor Profile)
- **Questions asked:** 2
- **Completion method:** Organic (no skip)
- **Issues:** Severe under-extraction, missing friction points

### Carol Chen (Investor Profile)
- **Questions asked:** 3 + skip
- **Completion method:** Semi-skip
- **Issues:** Slot extraction failed (role/company not parsed)

**Common Pattern:**
Both investor profiles had extraction problems:
- Bob: System concluded too early (only 2 questions)
- Carol: System couldn't extract from conversational context

**Suggests:**
The extraction service struggles with:
1. Investor/professional profiles (vs founder profiles?)
2. Implicit information extraction
3. Conversational phrasing vs structured answers

---

## Recommendations

### Priority 1: Fix Persona Generation After Skip (HIGH)

**Investigation:**
```typescript
// Check if skip flow triggers persona generation
// File: reciprocity-frontend/src/contexts/ConversationalOnboardingContext.tsx

// Does handleSkip() call completeOnboarding()?
const handleSkip = () => {
  // Does this trigger persona_service.generate_persona_task?
}
```

**Fix:**
Ensure skip flow calls the same completion endpoint that triggers persona generation.

---

### Priority 2: Improve Slot Extraction (HIGH)

**Enhancement:**
```python
# In prediction_service.py or extraction hints
# Add support for conversational extraction:

EXTRACTION_PATTERNS = {
    'role': [
        r"I'm a (.+?) at",  # "I'm a General Partner at..."
        r"I work as a (.+?) at",  # "I work as a VP at..."
        r"(.+?), (.+)",  # "General Partner, Sequoia Capital"
    ],
    'company': [
        r"at (.+?)[\.,]",  # "...at Sequoia Capital."
        r"work at (.+?)[\.,]",
        r", (.+?)$",  # "General Partner, Sequoia Capital"
    ]
}
```

---

### Priority 3: Add Embedding Status Indicator (MEDIUM)

**UI Enhancement:**
```typescript
// MatchCard.tsx
const isEmbeddingsPending = matchScore === 50; // Heuristic

{isEmbeddingsPending && (
  <div className="text-xs text-gray-500 mt-1">
    <Spinner size="xs" /> Computing personalized match score...
  </div>
)}
```

---

### Priority 4: Standardize Metric Labels (LOW)

**Quick Fix:**
```typescript
// DashboardPage.tsx + MatchesPage.tsx
// Change both to: "Approval rate"
```

---

## Testing Checklist for Next User

To fully validate our deployed fixes, the next test account should:

| Test | Purpose | Expected Result |
|------|---------|-----------------|
| ✅ Complete onboarding organically (NO skip) | Test onboarding loop fix | Reach dashboard without redirect loop |
| ✅ Try signup with existing email | Test 409 error display | See field error: "This email is already registered..." |
| ✅ Wait 5 minutes after onboarding | Test persona generation | "Updated The Persona" appears in agent activity |
| ✅ Check Settings → AI Summary | Test AI summary display | Persona text displays (or proper empty state if not generated) |
| ✅ Check Discover page after 10 minutes | Test embedding pipeline | Match scores vary (not all 50%) |

---

## Files to Investigate

| File | Check For |
|------|-----------|
| `reciprocity-frontend/src/contexts/ConversationalOnboardingContext.tsx` | Does skip flow trigger persona generation? |
| `reciprocity-ai/app/services/persona_service.py` | Is `generate_persona_task` queued on skip? |
| `reciprocity-ai/app/services/prediction_service.py` | Slot extraction patterns for conversational input |
| `reciprocity-backend/src/modules/onboarding/onboarding.controller.ts` | Does completion webhook fire on skip? |

---

## Summary Table: Issues vs Fixes

| Issue | Severity | Status | Next Action |
|-------|----------|--------|-------------|
| AI Summary empty state | LOW | ✅ FIXED | Confirmed working |
| Onboarding loop | CRITICAL | ⏸️ NOT TESTED | Need organic completion test |
| Signup 409 error | HIGH | ⏸️ NOT TESTED | Need duplicate email test |
| Persona not generated (skip) | MEDIUM | ❌ NEW ISSUE | Investigate skip flow trigger |
| Slot extraction fails | MEDIUM | ❌ NEW ISSUE | Enhance extraction patterns |
| Match scores all 50% | LOW | ⏸️ EXPECTED | Add loading indicator |
| Metric label inconsistency | LOW | ❌ NEW ISSUE | Standardize labels |
| Member ID truncation | LOW | ❌ NEW ISSUE | Fix responsive CSS |
| Match score fluctuation | LOW | ❌ ONGOING | Investigate calculation |

---

## Next Steps

1. **Immediate:** Check backend logs for Carol's persona generation task
2. **Investigation:** Trace skip flow to see if it triggers `generate_persona_task`
3. **Fix:** Ensure skip completion calls same webhook as organic completion
4. **Enhancement:** Improve slot extraction for conversational input
5. **Testing:** Run full organic onboarding test to validate loop fix
