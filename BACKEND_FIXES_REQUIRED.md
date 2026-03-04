# Backend Fixes Required - User Testing Issues
**Date:** 2026-03-03
**Priority:** HIGH 🔴

---

## Issue 4: Under-Extraction (Only 2 Questions) 🔴 CRITICAL

### Root Cause Found ✅

**File:** `app/services/progressive_disclosure.py`
**Line:** 352

```python
self.base_batch_size = int(os.getenv("DISCLOSURE_BATCH_SIZE", "2"))
```

**The Problem:**
- Default batch size is **2 questions per batch**
- Users are experiencing exactly this: only 2 questions during onboarding
- This is too conservative and makes onboarding feel incomplete

### Current Batch Size Logic

```python
# Line 352-354
self.base_batch_size = int(os.getenv("DISCLOSURE_BATCH_SIZE", "2"))  # DEFAULT: 2
self.max_batch_size = int(os.getenv("DISCLOSURE_MAX_BATCH", "4"))    # MAX: 4
self.min_batch_size = 1                                               # MIN: 1

# Line 421-434: Adaptive sizing based on engagement
def _calculate_batch_size(engagement, phase):
    if engagement == UserEngagementLevel.FRUSTRATED:
        return 1  # Show minimum questions
    elif engagement == UserEngagementLevel.LOW:
        return 1  # Show minimum questions
    elif engagement == UserEngagementLevel.HIGH:
        return min(3, 4)  # base_batch_size + 1 = 3
    else:  # MODERATE engagement (most common)
        return 2  # base_batch_size = 2 ⚠️ THIS IS THE PROBLEM
```

**What's Happening:**
1. Most users have MODERATE engagement by default
2. MODERATE engagement → returns `base_batch_size` = **2**
3. Users only see 2 questions per interaction
4. Onboarding takes too many turns to complete

---

## Fix: Increase Batch Sizes

### Recommended Configuration

**Change from:**
```python
self.base_batch_size = int(os.getenv("DISCLOSURE_BATCH_SIZE", "2"))
self.max_batch_size = int(os.getenv("DISCLOSURE_MAX_BATCH", "4"))
```

**Change to:**
```python
self.base_batch_size = int(os.getenv("DISCLOSURE_BATCH_SIZE", "5"))  # 2 → 5
self.max_batch_size = int(os.getenv("DISCLOSURE_MAX_BATCH", "7"))    # 4 → 7
```

### New Behavior After Fix

| Engagement Level | Questions Per Batch | Impact |
|------------------|---------------------|--------|
| FRUSTRATED | 1 | User wants to finish quickly (unchanged) |
| LOW | 1 | User giving short answers (unchanged) |
| MODERATE | **5** (was 2) | Default experience - most users |
| HIGH | **6** (was 3) | Engaged users get more questions |

### Implementation Options

**Option A: Environment Variable (Recommended - No Code Change)**
```bash
# Add to .env or deployment config
DISCLOSURE_BATCH_SIZE=5
DISCLOSURE_MAX_BATCH=7
```

**Option B: Change Default in Code**
```python
# File: app/services/progressive_disclosure.py
# Line 352-353

# Change from:
self.base_batch_size = int(os.getenv("DISCLOSURE_BATCH_SIZE", "2"))
self.max_batch_size = int(os.getenv("DISCLOSURE_MAX_BATCH", "4"))

# Change to:
self.base_batch_size = int(os.getenv("DISCLOSURE_BATCH_SIZE", "5"))
self.max_batch_size = int(os.getenv("DISCLOSURE_MAX_BATCH", "7"))
```

---

## Testing the Fix

### Before Fix
```
User: "I'm looking to raise funds for my startup"
AI: [Extracts 2 questions]
  1. What's your primary goal?
  2. Tell me about your background

Turn 2...
AI: [Extracts 2 more questions]
  3. What industry are you in?
  4. What stage is your company?

Turn 3...
[User frustrated - too many turns]
```

### After Fix
```
User: "I'm looking to raise funds for my startup"
AI: [Extracts 5 questions]
  1. What's your primary goal?
  2. Tell me about your background
  3. What industry are you in?
  4. What stage is your company?
  5. What's your funding goal?

Turn 2...
[Onboarding complete or nearly complete]
```

### Test Script

1. **Set environment variable:**
   ```bash
   export DISCLOSURE_BATCH_SIZE=5
   export DISCLOSURE_MAX_BATCH=7
   ```

2. **Start fresh onboarding session:**
   ```bash
   POST /onboarding/start
   {
     "user_id": "test_user_123",
     "objective": "fundraising"
   }
   ```

3. **Send rich message:**
   ```bash
   POST /onboarding/chat
   {
     "user_id": "test_user_123",
     "message": "I'm Jane Doe, a marketing director at TechCorp. I'm looking to expand my B2B network and find potential partners for our SaaS product. I prefer LinkedIn for outreach and value quality over quantity."
   }
   ```

4. **Verify response has 5+ extracted slots:**
   ```json
   {
     "extracted_slots": {
       "name": "Jane Doe",
       "role": "marketing director",
       "company": "TechCorp",
       "primary_goal": "expand B2B network",
       "industry_focus": "SaaS/B2B",
       "engagement_style": "quality over quantity"
     },
     "next_questions": ["...", "...", "...", "...", "..."],  // 5 questions
     "completion_percent": 65.0  // Should be higher now
   }
   ```

---

## Issue 3: AI Summary Inconsistency ⚠️ NEEDS INVESTIGATION

### Observed Behavior

**Carol Chen (Test 3):**
- Saw proper empty state: "No AI summary available yet..."

**Bob Baker (Test 4):**
- Saw completely blank content

### Current Backend Logic ✅ (No Obvious Bugs)

**File:** `reciprocity-backend/src/modules/profile/profile.service.ts`
**Line:** 189-210

```typescript
async getSummary(userId: string) {
  return this.sequelize.transaction(async t => {
    let summaryRecord = await this.userSummaryModel.findOne({
      where: { user_id: userId },
      attributes: ['id', 'summary', 'status', 'version', 'webhook'],
      order: [['created_at', 'DESC']],
      transaction: t,
    });

    if (summaryRecord) {
      summaryRecord.summary = JSON.parse(summaryRecord.summary);
    }
    return summaryRecord;
  });
}
```

**Logic is correct.** The issue is likely:

### Possible Root Causes

1. **Different `webhook` values**
   - Carol: `{ webhook: true, summary: "" }` → Frontend shows proper empty state
   - Bob: `{ webhook: false }` OR `{ webhook: null }` → Frontend shows loader indefinitely

2. **Different `summary` content**
   - Carol: `summary: ""` (empty string)
   - Bob: `summary: null` (null value)

3. **Race condition in WebSocket flow**
   ```
   WebSocket: "summary.ready" fires
   → Frontend calls refetch()
   → Backend hasn't updated database yet
   → Returns { webhook: false } or null
   → Frontend shows blank/loader
   ```

4. **Silent failure in persona generation**
   - Pipeline: `process_resume_task → generate_persona_task`
   - If `generate_persona_task` fails silently, `webhook` never set to `true`
   - Database has record but `summary` is empty/null

---

## Investigation Steps for Issue 3

### 1. Check Database Consistency

**Query the UserSummaries table:**
```sql
SELECT
  user_id,
  summary,
  status,
  webhook,
  created_at
FROM user_summaries
WHERE user_id IN ('carol_chen_id', 'bob_baker_id')
ORDER BY created_at DESC
LIMIT 2;
```

**Expected Patterns:**
- **Carol:** `webhook: true`, `summary: ""` or `summary: null`
- **Bob:** `webhook: false/null`, `summary: null`

---

### 2. Check Persona Generation Task

**File:** `reciprocity-ai/app/tasks/persona_generation.py` (or similar)

**Questions:**
1. Does `generate_persona_task` always set `webhook: true` on completion?
2. Are there silent failures (try/except without logging)?
3. Is there proper error handling for Gemini API failures?

**Look for:**
```python
@celery_app.task
def generate_persona_task(user_id: str):
    try:
        # Generate persona
        persona = generate_persona(user_id)

        # ⚠️ CHECK: Is webhook ALWAYS set to true here?
        update_summary(user_id, persona, webhook=True)

        # ⚠️ CHECK: Is WebSocket event sent AFTER database update?
        send_webhook("summary.ready", user_id)

    except Exception as e:
        # ⚠️ CHECK: Silent failures?
        logger.error(f"Persona generation failed: {e}")
        # Should still set webhook: false to prevent infinite loader
```

---

### 3. Check WebSocket Event Timing

**File:** Location where `summary.ready` event is sent

**Race condition check:**
```python
# ❌ BAD: Send event before database update
send_webhook("summary.ready", user_id)
update_summary_in_db(user_id, summary)

# ✅ GOOD: Send event after database update
update_summary_in_db(user_id, summary)
send_webhook("summary.ready", user_id)
```

---

### 4. Add Logging to Track the Issue

**Add to `reciprocity-backend/src/modules/profile/profile.service.ts`:**

```typescript
async getSummary(userId: string) {
  this.logger.log(`----- GET SUMMARY -----`);
  this.logger.log({ user_id: userId });

  return this.sequelize.transaction(async t => {
    let summaryRecord = await this.userSummaryModel.findOne({
      where: { user_id: userId },
      attributes: ['id', 'summary', 'status', 'version', 'webhook'],
      order: [['created_at', 'DESC']],
      transaction: t,
    });

    // ADD LOGGING HERE
    if (!summaryRecord) {
      this.logger.warn(`No summary record found for user ${userId}`);
    } else {
      this.logger.log(`Summary found: webhook=${summaryRecord.webhook}, summary_length=${summaryRecord.summary?.length || 0}`);
    }

    if (summaryRecord) {
      summaryRecord.summary = JSON.parse(summaryRecord.summary);
    }
    return summaryRecord;
  });
}
```

---

## Recommended Fixes for Issue 3

### Fix 1: Ensure `webhook` is Always Set

**File:** Persona generation task

```python
@celery_app.task
def generate_persona_task(user_id: str):
    try:
        persona = generate_persona(user_id)

        # Set webhook: true on success
        update_summary(user_id, persona, webhook=True)

        # Send WebSocket event AFTER database update
        send_webhook("summary.ready", user_id)

    except Exception as e:
        logger.error(f"Persona generation failed for {user_id}: {e}")

        # IMPORTANT: Set webhook: false to prevent infinite loader
        update_summary(user_id, summary=None, webhook=False)

        # Send failed event so frontend can show error
        send_webhook("summary.failed", user_id)
```

### Fix 2: Frontend Timeout (Already Implemented ✅)

The frontend fix we already implemented handles this:
```tsx
// If webhook: true but summary is empty → Show proper empty state
if (data?.result && data?.result?.webhook && !data?.result?.summary) {
  return <EmptyState />
}
```

---

## Priority and Timeline

| Issue | Priority | Estimated Time | Owner |
|-------|----------|----------------|-------|
| Issue 4: Batch Size | 🔴 HIGH | 5 minutes (env var) OR 10 minutes (code change) | Backend Team |
| Issue 3: Investigation | ⚠️ MEDIUM | 2-4 hours | AI Service Team |

---

## Deployment Steps

### For Issue 4 (Batch Size)

**Option A: Environment Variable (Recommended)**
1. Add to `.env` or deployment config:
   ```bash
   DISCLOSURE_BATCH_SIZE=5
   DISCLOSURE_MAX_BATCH=7
   ```
2. Restart AI service
3. Test with new onboarding session

**Option B: Code Change**
1. Edit `app/services/progressive_disclosure.py` line 352-353
2. Change defaults from "2"/"4" to "5"/"7"
3. Commit and deploy
4. Test with new onboarding session

### For Issue 3 (Summary Consistency)

1. Run database query to compare Carol vs Bob records
2. Check persona generation task logs for failures
3. Add logging to track webhook setting
4. Test with 2 fresh users to reproduce
5. Implement fixes based on findings

---

## Success Criteria

### Issue 4 Fix Success
- [ ] Environment variable set OR code changed
- [ ] AI service restarted
- [ ] New onboarding session extracts 5+ questions
- [ ] `completion_percent` reaches 70%+ after first rich message
- [ ] Onboarding completes in 2-3 turns instead of 5-7

### Issue 3 Fix Success
- [ ] Database query shows consistent `webhook` values
- [ ] Persona generation task logs show no silent failures
- [ ] 2 test users both see consistent empty state OR proper content
- [ ] No more blank screens reported

---

## Rollback Plan

### Issue 4 Rollback
If users complain about too many questions:
```bash
# Revert to original values
DISCLOSURE_BATCH_SIZE=2
DISCLOSURE_MAX_BATCH=4
```

### Issue 3 Rollback
Frontend fix is non-breaking and can stay. Backend investigation has no code changes to rollback.

---

## Next Steps

1. **IMMEDIATE:** Set `DISCLOSURE_BATCH_SIZE=5` and `DISCLOSURE_MAX_BATCH=7` in production
2. **TODAY:** Test Issue 4 fix with 2-3 new users
3. **THIS WEEK:** Complete Issue 3 investigation (database query + task logs)
4. **THIS WEEK:** Implement Issue 3 fixes based on findings

---

## Contact for Questions

- **Issue 4 (Batch Size):** AI Service Team
- **Issue 3 (Summary):** Backend Team + AI Service Team
- **Testing:** QA Team

---

## Related Documents

- [USER_TESTING_ISSUES_INVESTIGATION.md](../reciprocity-frontend/USER_TESTING_ISSUES_INVESTIGATION.md) - Frontend fixes (already deployed)
- User Test 3: Carol Chen analysis
- User Test 4: Bob Baker full test analysis
