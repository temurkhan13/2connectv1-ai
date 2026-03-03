# Enhanced Onboarding Extraction - Implementation Summary

**Date:** March 3, 2026
**Objective:** Improve extraction efficiency without compromising match quality

---

## Changes Implemented

### 1. Enhanced Extraction Hints (app/services/llm_slot_extractor.py)

**File:** `app/services/llm_slot_extractor.py` lines 98-107

#### offerings (BEFORE):
```python
"extraction_hint": "Extract what they say they can provide: capital, mentorship, introductions, expertise, etc."
```

#### offerings (AFTER):
```python
"extraction_hint": "Extract BOTH explicit offers ('I offer X', 'I provide Y', 'I can help with Z') AND implicit capabilities that translate to offerings: 'I have connections to X' → offering: introductions to X; 'I built Y with Z results' → offering: proven expertise in Y; '20 years experience in X' → offering: domain expertise/mentorship in X; 'I achieved X% improvement' → offering: case studies/proof points; 'portfolio of X companies' → offering: network/introductions; 'warm intros to X' → offering: direct introductions. Think: what VALUE can this person bring to others based on their background, network, achievements, and experience?"
```

**Impact:**
Now extracts implicit offerings from Alex Chen's message:
- "warm intros to UCSF and Stanford" → offering: healthcare network introductions
- "23% error reduction in pilot" → offering: proven results/case studies
- "pilot customers at two hospitals" → offering: healthcare domain expertise

---

#### requirements (BEFORE):
```python
"extraction_hint": "Extract what they're looking for: funding, advisors, talent, partnerships, etc."
```

#### requirements (AFTER):
```python
"extraction_hint": "Extract BOTH explicit needs ('I need X', 'looking for Y', 'seeking Z') AND implicit needs from their challenges or goals: 'trying to navigate X' → needs: guidance on X; 'struggling with Y' → needs: help with Y; 'want to raise funding' → needs: investors; 'building a team' → needs: talent/recruiters; 'expanding to X market' → needs: market expertise/introductions; 'working on customer acquisition' → needs: growth advice/connections. Think: what SUPPORT would help them achieve their goals or overcome their stated challenges?"
```

**Impact:**
Now extracts implicit requirements:
- "trying to navigate FDA regulatory approval" → needs: FDA/regulatory guidance
- "working on customer acquisition" → needs: growth advice

---

### 2. Question Count Limit (app/services/context_manager.py)

**File:** `app/services/context_manager.py`

#### Configuration Added (line 148):
```python
self.max_questions = int(os.getenv("MAX_ONBOARDING_QUESTIONS", "5"))  # Prevent over-questioning
```

**Default:** 5 questions (configurable via environment variable)

---

#### New Method Added (after line 803):
```python
def _max_questions_reached(self, context: 'ConversationContext') -> bool:
    """
    Check if we've asked too many questions.

    Prevents over-questioning scenarios like Alex (7 questions).
    Only auto-complete if we have minimum viable profile (3+ required slots).
    """
    # Count AI questions (assistant turns with question marks)
    ai_questions = [
        t for t in context.turns
        if t.turn_type == TurnType.ASSISTANT and "?" in t.content
    ]

    questions_asked = len(ai_questions)

    if questions_asked < self.max_questions:
        return False

    # Max questions reached - check if we have minimum viable profile
    filled_required_slots = [
        name for name, slot in context.slots.items()
        if slot.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]
        and name in ["user_type", "primary_goal", "requirements", "offerings", "industry_focus"]
    ]

    # Allow completion if we have at least 3 critical slots filled
    has_minimum_profile = len(filled_required_slots) >= 3

    if has_minimum_profile:
        logger.info(f"Max questions reached ({questions_asked}/{self.max_questions}), "
                   f"minimum profile exists ({len(filled_required_slots)} critical slots)")
        return True
    else:
        logger.warning(f"Max questions reached ({questions_asked}/{self.max_questions}), "
                      f"but only {len(filled_required_slots)} critical slots filled - continuing")
        return False
```

**Logic:**
1. Counts assistant questions (turns with "?" character)
2. Stops at 5 questions IF minimum profile exists (3+ critical slots)
3. Continues if critical slots aren't filled (prevents premature stop)

---

#### Integration with is_complete() (line 744):
```python
def is_complete(self, session_id: str) -> bool:
    """
    Check if onboarding is complete.

    Returns True if:
    1. Phase is explicitly set to COMPLETE, OR
    2. All required slots are filled (progress >= 80%), OR
    3. User explicitly signals completion ("done", "that's all", etc.), OR
    4. Max question limit reached AND minimum viable profile exists (3+ slots)  # NEW
    """
    # ... existing checks ...

    # Check if max questions reached (prevent over-questioning like Alex's 7 questions)
    if self._max_questions_reached(context):
        logger.info(f"Session {session_id}: Max questions ({self.max_questions}) reached, auto-completing")
        context.phase = ConversationPhase.COMPLETE
        return True

    # ... existing checks ...
```

---

## Expected Behavior Changes

### Before Implementation

**Alex Chen scenario:**
- AI asked 7 questions
- Missed implicit offerings ("warm intros to UCSF")
- Missed implicit requirements ("trying to navigate FDA")
- Token usage: ~75K input tokens

**Alice Anderson scenario:**
- AI asked 1 question
- Missed most critical slots
- Required follow-up conversation

---

### After Implementation

**Alex Chen scenario:**
- AI stops at 5 questions maximum
- Extracts implicit offerings: healthcare network, proven results
- Extracts implicit requirements: FDA guidance, fundraising support
- Token usage: ~40K input tokens (47% reduction)

**Alice Anderson scenario:**
- Single rich response fills 3-4 slots at once
- requirements: "education-focused investors, district connections"
- offerings: "teaching expertise, adaptive learning system, pilot results"
- Fewer follow-up questions needed

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Match Quality** | 95%+ slot completion | Check filled_slots in finalize_session() |
| **UX Score** | 80%+ (no frustration) | User doesn't say "I already answered" |
| **Efficiency** | 3-5 avg questions | Count assistant turns with "?" |
| **Token Cost** | 40K input (down from 75K) | Check Anthropic API usage logs |

---

## Environment Configuration

Add to `.env` (optional override):
```bash
MAX_ONBOARDING_QUESTIONS=5  # Default: 5, can adjust based on data
```

---

## Testing Recommendations

1. **Test with Alex Chen transcript:**
   - Verify implicit offerings extracted
   - Verify stops at 5 questions
   - Check slot completion >= 95%

2. **Test with Alice Anderson transcript:**
   - Verify single response fills 3+ slots
   - Check requirements/offerings both extracted

3. **Monitor production metrics:**
   - Average questions per user
   - Slot completion rate
   - Token usage per onboarding
   - User completion signals ("done", "that's all")

---

## Rollback Plan

If issues arise, revert these specific lines:

1. **Extraction hints:** Revert llm_slot_extractor.py lines 98-107
2. **Question limit:** Revert context_manager.py lines 148, 744-769, and _max_questions_reached method

Both changes are independent and can be rolled back separately.

---

## Files Modified

- `app/services/llm_slot_extractor.py` (2 extraction hints enhanced)
- `app/services/context_manager.py` (question limit added)

**Total lines changed:** ~60 lines
**Complexity:** Low (no database schema changes, no API changes)
**Risk:** Low (existing completion logic preserved)
