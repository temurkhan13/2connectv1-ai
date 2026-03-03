-- =============================================================================
-- CREATE onboarding_answers TABLE - CONVERSATIONAL SLOT STORAGE
-- =============================================================================
-- This table stores LLM-extracted slots from chat-based onboarding
-- Separate from user_onboarding_answers (form-based, in backend PostgreSQL)
--
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard/project/omcjxrhprhtlwqzuhjqb/sql
-- =============================================================================

-- 1. Create onboarding_answers table
CREATE TABLE IF NOT EXISTS onboarding_answers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    slot_name TEXT NOT NULL,
    value TEXT,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    source_text TEXT,
    extraction_method TEXT DEFAULT 'llm',  -- 'llm' or 'regex'
    status TEXT DEFAULT 'filled',  -- 'filled', 'confirmed', 'skipped'
    created_at TIMESTAMPTZ DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    updated_at TIMESTAMPTZ DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),

    -- Ensure one slot per user (upsert support)
    UNIQUE(user_id, slot_name)
);

-- 2. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_onboarding_answers_user_id ON onboarding_answers(user_id);
CREATE INDEX IF NOT EXISTS idx_onboarding_answers_slot_name ON onboarding_answers(slot_name);
CREATE INDEX IF NOT EXISTS idx_onboarding_answers_status ON onboarding_answers(status);
CREATE INDEX IF NOT EXISTS idx_onboarding_answers_created_at ON onboarding_answers(created_at DESC);

-- 3. Create trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_onboarding_answers_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP AT TIME ZONE 'UTC';
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_onboarding_answers_updated_at
    BEFORE UPDATE ON onboarding_answers
    FOR EACH ROW
    EXECUTE FUNCTION update_onboarding_answers_updated_at();

-- 4. Create helper function to get user's filled slots count
CREATE OR REPLACE FUNCTION get_user_slots_count(p_user_id TEXT)
RETURNS INTEGER AS $$
    SELECT COUNT(*)::INTEGER
    FROM onboarding_answers
    WHERE user_id = p_user_id
      AND status IN ('filled', 'confirmed');
$$ LANGUAGE SQL STABLE;

-- 5. Create helper function to get user's slots as JSON
CREATE OR REPLACE FUNCTION get_user_slots_json(p_user_id TEXT)
RETURNS JSONB AS $$
    SELECT COALESCE(
        jsonb_object_agg(slot_name, jsonb_build_object(
            'value', value,
            'confidence', confidence,
            'status', status,
            'created_at', created_at
        )),
        '{}'::jsonb
    )
    FROM onboarding_answers
    WHERE user_id = p_user_id;
$$ LANGUAGE SQL STABLE;

-- 6. Grant permissions (adjust if RLS enabled)
-- GRANT ALL ON onboarding_answers TO authenticated;
-- GRANT ALL ON onboarding_answers TO anon;
-- GRANT EXECUTE ON FUNCTION get_user_slots_count TO authenticated;
-- GRANT EXECUTE ON FUNCTION get_user_slots_json TO authenticated;

-- 7. Verify table was created
DO $$
BEGIN
    RAISE NOTICE 'onboarding_answers table created successfully!';
    RAISE NOTICE 'Helper functions: get_user_slots_count(), get_user_slots_json()';
END $$;

SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'onboarding_answers'
ORDER BY ordinal_position;
