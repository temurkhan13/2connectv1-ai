# Safe Fixes Plan — 2Connect Platform Issues
**Date:** March 3, 2026
**Context:** User testing identified real UX issues (excluding onboarding loop = bad credentials)

---

## Issues to Fix (Prioritized by Safety & Impact)

### ✅ SAFE TO FIX (No Risk to Core Functionality)

#### 1. Signup 409 Error - No User Feedback (MEDIUM, 15 min)
**Problem:** When email already exists, HTTP 409 returned but no error message shown. Form appears frozen.

**Fix Location:** `reciprocity-frontend/src/pages/Auth/SignupPage.tsx` (or similar)

**Solution:**
```typescript
// Add error handling in signup form
try {
  await signup(email, password, name);
  navigate('/onboarding/chat');
} catch (error) {
  if (error.response?.status === 409) {
    setError("This email is already registered. Please log in instead.");
    // Optional: Show "Go to Login" link
  } else {
    setError("Signup failed. Please try again.");
  }
}
```

**Impact:** ✅ High (improves signup UX), ⚠️ Risk: None (error handling only)

---

#### 2. AI Summary Page Renders Blank (MEDIUM, 20 min)
**Problem:** Settings → AI Summary shows empty text box despite persona being generated on backend.

**Fix Location:** `reciprocity-frontend/src/pages/Settings/AISummary.tsx` (or similar)

**Root Cause Options:**
1. Frontend not fetching persona from correct endpoint
2. Persona data exists but not being passed to textarea value
3. API response structure mismatch

**Investigation Steps:**
```typescript
// Check if persona fetch is working
const { data: persona, isLoading, error } = useQuery(['persona'], fetchPersona);

console.log('Persona data:', persona);  // Debug

// Check textarea binding
<textarea
  value={persona?.summary || persona?.content || ''}  // Try multiple fields
  readOnly
/>
```

**Solution:**
- Fix API endpoint call if wrong
- Add proper data mapping from backend response
- Add loading state and error message

**Impact:** ✅ High (enables viewing generated persona), ⚠️ Risk: Low (read-only feature)

---

#### 3. Message Streaming Timeout (HIGH, 30 min)
**Problem:** Welcome message takes 80+ seconds to stream (Report 1 only, may be network/API issue)

**Fix Location:** `reciprocity-ai/app/routers/onboarding.py` (streaming endpoint)

**Investigation:**
```python
# Check current streaming implementation
async def stream_ai_response(message: str):
    response = await llm.generate(message)

    # Check chunk size and delay
    for chunk in response:
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.05)  # Too long?
```

**Optimization:**
- Reduce chunk delay from 0.05s → 0.01s (5x faster)
- Increase chunk size (send 5-10 words per chunk instead of 1-2)
- Add timeout monitoring (log if >5s for first chunk)

**Impact:** ✅ High (first impression), ⚠️ Risk: Medium (streaming is complex, test thoroughly)

---

### ⚠️ INVESTIGATE BEFORE FIXING (Requires Understanding Root Cause)

#### 4. Inconsistent Match Score Metrics (MEDIUM, 45 min)
**Problem:** Dashboard shows "11-50%" (fluctuates), Matches page shows "7%", different page loads show different values.

**Possible Causes:**
1. Different calculation methods (Dashboard = all users, Matches = matched only)
2. Caching issue (stale data)
3. Race condition in async data fetch
4. Frontend displaying wrong field from API response

**Investigation Steps:**
```bash
# 1. Check API responses
curl https://twoconnectv1-backend.onrender.com/api/v1/dashboard/stats
curl https://twoconnectv1-backend.onrender.com/api/v1/matches

# 2. Compare calculation logic
# Dashboard: app/modules/dashboard/dashboard.service.ts
# Matches: app/modules/matches/matches.service.ts

# 3. Check if metrics are computed differently
```

**Do NOT fix until root cause understood** - could break actual matching algorithm

---

#### 5. Profile Review Modal Inconsistently Shown (MEDIUM, 1 hr)
**Problem:** 3/5 users got modal, 2/5 skipped. Logic unclear.

**Investigation:**
```typescript
// Find where modal is triggered
// reciprocity-frontend/src/components/ProfileReviewModal.tsx (if exists)

// Check conditions:
// - Is it random?
// - Based on slot completion percentage?
// - Based on user type?
// - A/B test flag?
```

**Do NOT fix until** we understand if this is intentional (A/B test) or bug

---

### 🚫 DO NOT FIX (Out of Scope or Expected Behavior)

#### ❌ No AI-Generated Matches After 1 Hour
**Reason:** Expected behavior for brand new users. Scheduled matching task runs every 15 min but may need more users in pool.

#### ❌ WebSocket 429 Rate Limit on Initial Load
**Reason:** Known issue per test notes. External rate limit from Stream Chat service. Resolves on refresh.

#### ❌ Occasional "Failed to Send Message" Error
**Reason:** Network/backend timeout. Needs retry logic but not critical (low frequency).

---

## Implementation Order (Recommended)

### Phase 1: Quick Wins (1 hour total)
1. ✅ **Signup 409 error message** (15 min) - Frontend only, no risk
2. ✅ **AI Summary blank page** (20 min) - Frontend data binding
3. ✅ **Message streaming optimization** (30 min) - Backend tuning

### Phase 2: Investigations (2 hours)
4. ⚠️ **Investigate match score inconsistency** (45 min) - Understand before fixing
5. ⚠️ **Investigate profile review modal logic** (1 hr) - Check if intentional

---

## Files to Modify

### Frontend (reciprocity-frontend)
| File | Change | Risk |
|------|--------|------|
| `src/pages/Auth/SignupPage.tsx` | Add 409 error handling | None |
| `src/pages/Settings/AISummary.tsx` | Fix persona data binding | Low |
| `src/hooks/useAuth.ts` (if signup logic here) | Error state management | Low |

### Backend (reciprocity-ai)
| File | Change | Risk |
|------|--------|------|
| `app/routers/onboarding.py` | Optimize streaming chunk delay | Medium |
| `app/services/context_manager.py` | Already fixed (question limit) | None |
| `app/services/llm_slot_extractor.py` | Already fixed (extraction hints) | None |

### Investigations Only (Do Not Modify Yet)
| File | Purpose |
|------|---------|
| `reciprocity-backend/src/modules/dashboard/dashboard.service.ts` | Check match score calculation |
| `reciprocity-backend/src/modules/matches/matches.service.ts` | Compare with dashboard calculation |
| `reciprocity-frontend/src/components/ProfileReviewModal.tsx` | Understand modal trigger logic |

---

## Testing Checklist

### After Each Fix
- [ ] **Signup 409:** Test with existing email → error message appears
- [ ] **AI Summary:** Complete onboarding → Settings → AI Summary shows text
- [ ] **Streaming:** New user onboarding → welcome message appears <10s

### Regression Testing
- [ ] **Onboarding completion:** Still works after streaming changes
- [ ] **Persona generation:** Still triggers after finalization
- [ ] **Dashboard:** Still loads after onboarding
- [ ] **Match scores:** Still calculated correctly

---

## Safety Guidelines

### ✅ Safe Changes (Proceed)
- Error message additions (no logic changes)
- UI data binding fixes (read-only features)
- Performance tuning (delay reduction with fallback)

### ⚠️ Risky Changes (Test Thoroughly)
- Streaming logic (affects real-time UX)
- Metric calculations (affects matching algorithm perception)
- Modal trigger logic (may break onboarding flow)

### 🚫 Do Not Touch
- Matching algorithm (`matching_service.py`, `postgresql.py find_similar_users`)
- Persona generation pipeline (already working)
- Onboarding state management (already working, was test artifact)
- Embedding generation (working, inferred from scores)

---

## Rollback Plan

Each fix is independent and can be reverted separately:

**Signup 409 Error:**
```bash
git revert <commit-hash>  # Remove error handling only
```

**AI Summary Fix:**
```bash
git revert <commit-hash>  # Revert data binding changes
```

**Streaming Optimization:**
```bash
# Revert streaming delay change
# Or: Set env var STREAM_CHUNK_DELAY=0.05 (restore old value)
```

---

## Success Criteria

| Fix | Metric | Target |
|-----|--------|--------|
| Signup 409 | Error message visible | 100% of attempts |
| AI Summary | Persona text renders | 100% after generation |
| Streaming | Welcome message time | <10 seconds |

---

## Next Steps

**Option A: Implement Quick Wins Now (Recommended)**
- Fix #1 (signup error), #2 (AI summary), #3 (streaming)
- Test on staging
- Deploy if tests pass

**Option B: Investigate First**
- Reproduce each issue on staging
- Confirm root causes
- Then implement fixes

Which approach do you prefer?
