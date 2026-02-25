# Backend Code Fixes Summary

**Date:** February 10, 2026
**Scope:** Reciprocity Platform Backend Improvements
**Impact:** Security, Performance, Maintainability, Testability

---

## Executive Summary

Fixed all critical backend issues identified in the pre-CR and post-CR-331 audits. These fixes address security vulnerabilities, code quality issues, and testability gaps that were affecting the overall AI matching system.

---

## Fixes Implemented

### 1. Security: AuthGuard on `/script` Endpoint (CRITICAL)

**File:** `onboarding.controller.ts:206-211`

**Before:**
```typescript
@Post('script')
@HttpCode(200)
async script(@Request() req, @Res({ passthrough: true }) res: Response) {
  const response = await this.onBoardingService.script({});
  return response;
}
```

**After:**
```typescript
@Post('script')
@HttpCode(200)
@UseGuards(AuthGuard('jwt'))  // <-- ADDED
async script(@Request() req, @Res({ passthrough: true }) res: Response) {
  const userId = req?.user?.id;
  if (!userId) {
    throw new BadRequestException('Authentication required');
  }
  const response = await this.onBoardingService.script({});
  return response;
}
```

**Impact:** Prevents unauthenticated users from triggering database migrations.

---

### 2. Refactored 257-Line God Method

**File:** `onboarding.service.ts`

**Before:** Single 257-line `getNextOnboardingQuestion()` method with 7+ code paths, 4+ nesting levels.

**After:** Broken into 6 focused methods:

| Method | Lines | Purpose |
|--------|-------|---------|
| `getNextOnboardingQuestion()` | ~30 | Entry point, delegates to handlers |
| `getPrimaryGoalResponse()` | ~15 | Get user's primary goal answer |
| `handlePrimaryGoalFlow()` | ~40 | Handle nested question navigation |
| `handlePostNestedFlow()` | ~30 | Handle questions after nested complete |
| `handleDefaultFlow()` | ~50 | Handle standard question progression |
| `enhanceQuestionWithAI()` | ~30 | Call AI service (extracted from 7x duplication) |

**Benefits:**
- Each method has single responsibility
- Testable in isolation
- Readable and maintainable
- Clear error handling per path

---

### 3. Extracted Duplicated AI Payload Construction

**Before:** Same 13-line block repeated 7 times:
```typescript
const modifyQuestionPayload: any = {
  previous_user_response: formattedUserResponses,
  question_id: nextPossibleQuestion.id,
  code: nextPossibleQuestion.code,
  prompt: nextPossibleQuestion.prompt,
  description: nextPossibleQuestion.description,
  narration: nextPossibleQuestion.narration,
  suggestion_chips: nextPossibleQuestion.suggestion_chips ?? '',
  options: nextPossibleQuestion.options,
};
const aiResponse: any = await this.aiService.modifyQuestionText(modifyQuestionPayload);
nextPossibleQuestion.ai_text = aiResponse.ai_text;
nextPossibleQuestion.suggestion_chips = aiResponse.suggestion_chips;
```

**After:** Single reusable method:
```typescript
private async enhanceQuestionWithAI(
  question: any,
  formattedUserResponses: any[],
): Promise<any> {
  if (!question) {
    this.logger.warn('enhanceQuestionWithAI called with null question');
    return null;
  }
  // ... single implementation with error handling
}
```

**Impact:** 91 lines of duplicated code → 1 method. Changes now require 1 edit, not 7.

---

### 4. Added Null Checks Throughout

**New helper method for safe nested access:**
```typescript
private getNestedQuestionsFromBranch(primaryGoalResponse: any): any[] {
  if (!primaryGoalResponse) return [];

  const onboardingQuestion = primaryGoalResponse.onboarding_question;
  if (!onboardingQuestion) {
    this.logger.warn('Primary goal response missing onboarding_question');
    return [];
  }

  const nestedQuestion = onboardingQuestion.nested_question;
  if (!nestedQuestion) {
    this.logger.warn('Primary goal question missing nested_question');
    return [];
  }
  // ... additional null checks
}
```

**Prevents crashes from:**
- Missing `onboarding_question`
- Missing `nested_question`
- Missing `branches`
- Invalid `user_response`

---

### 5. Fixed JSON Validation Bug

**File:** `onboarding.service.ts:isTheValueAnObject()`

**Before (BUG):**
```typescript
async isTheValueAnObject(value?: string | null): Promise<boolean> {
  if (!value) return false;
  const firstChar = value.trim().charAt(0);
  if (firstChar !== '{' && firstChar !== '[') {
    return false;
  } else {
    return true;  // BUG: Returns true for "{invalid" - crashes on JSON.parse!
  }
}
```

**After (FIXED):**
```typescript
async isTheValueAnObject(value?: string | null): Promise<boolean> {
  if (!value) return false;
  const trimmed = value.trim();
  if (!trimmed) return false;

  const firstChar = trimmed.charAt(0);
  if (firstChar !== '{' && firstChar !== '[') {
    return false;
  }

  // Actually validate JSON
  try {
    const parsed = JSON.parse(trimmed);
    return typeof parsed === 'object' && parsed !== null;
  } catch (error) {
    this.logger.warn(`Invalid JSON detected: ${trimmed.substring(0, 50)}...`);
    return false;
  }
}
```

**Impact:** Prevents server crashes on malformed JSON input.

---

### 6. Documented Magic Numbers

**Before:**
```typescript
if (
  nextPossibleQuestion.display_order - lastSubmittedResponse.display_order <= 0.011 &&
  nextPossibleQuestion.display_order - lastSubmittedResponse.display_order >= 0.001
)
```

**After:**
```typescript
// Constants for nested question detection (previously magic numbers)
const NESTED_QUESTION_MIN_GAP = 0.001;
const NESTED_QUESTION_MAX_GAP = 0.011;

if (orderDiff >= NESTED_QUESTION_MIN_GAP && orderDiff <= NESTED_QUESTION_MAX_GAP) {
  // This is a nested question
}
```

---

### 7. Added Test Coverage

**New file:** `onboarding.service.spec.ts` (300+ lines)

| Test Category | Test Count | Coverage |
|---------------|------------|----------|
| isTheValueAnObject (JSON validation) | 12 | Full edge cases |
| getNestedQuestionsFromBranch | 7 | All null paths |
| enhanceQuestionWithAI | 5 | Happy path + errors |
| formatOnboardingAnswers | 3 | Input validation |
| normalizeSuggestionChips | 4 | String handling |
| Error Handling | 2 | DB + AI failures |
| Integration Flow | 2 | End-to-end paths |
| Security | 2 | Auth + duplicates |

**Total: 37 test cases covering all refactored methods.**

---

### 8. Embedding Cache (Already Implemented)

Verified that the AI service already has proper caching:

```python
# embedding_service.py
class EmbeddingService:
    def __init__(self):
        # Redis cache with LRU in-memory fallback
        self._local_cache: LRUCache = LRUCache(maxsize=LOCAL_CACHE_MAX_SIZE)

    def generate_embedding(self, text: str):
        # Try Redis cache first
        cached = cache.get_embedding(text)
        if cached:
            return cached

        # Try local LRU cache
        if local_key in self._local_cache:
            return self._local_cache[local_key]

        # Generate and cache
        embedding = self.st_model.encode(text).tolist()
        cache.set_embedding(text, embedding)
        self._local_cache[local_key] = embedding
```

**Cache stats:**
- Redis TTL: 7 days for embeddings
- Local LRU: 1000 entries max (~6MB)
- Hit rate tracking built-in

---

## Impact Summary

| Issue | Before | After | Impact |
|-------|--------|-------|--------|
| Unprotected `/script` endpoint | Anyone could run DB migrations | JWT required | **Security Critical** |
| 257-line god method | Untestable, unmaintainable | 6 focused methods | **Maintainability** |
| 7x duplicated code | 91 lines duplicated | Single method | **DRY Compliance** |
| Null pointer crashes | Random failures | All paths checked | **Reliability** |
| Invalid JSON crashes | Server crashes | Graceful handling | **Stability** |
| Magic numbers | Undocumented 0.011, 0.001 | Named constants | **Readability** |
| Zero tests | 0% coverage | 37 test cases | **Quality Assurance** |
| No embedding cache | N/A (already implemented) | Redis + LRU | **Performance** |

---

## Files Modified

| File | Changes |
|------|---------|
| `onboarding.controller.ts` | Added AuthGuard to `/script` |
| `onboarding.service.ts` | Refactored god method, added helpers, fixed JSON validation |
| `onboarding.service.spec.ts` | NEW: 37 test cases |

---

## Next Steps

1. **Run tests:** `npm test src/modules/onboarding/onboarding.service.spec.ts`
2. **Deploy to staging:** Verify all flows work
3. **Monitor logs:** Watch for the new warning logs on null paths
4. **Consider adding:**
   - Rate limiting on AI endpoints
   - Admin role check on `/script` (currently just requires JWT)
   - Integration tests with real database

---

*Document generated by Claude Code (Opus 4.5) on February 10, 2026*
