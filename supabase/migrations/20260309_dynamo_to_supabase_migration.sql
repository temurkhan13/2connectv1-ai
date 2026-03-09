-- =============================================================================
-- Migration: DynamoDB to Supabase
-- Date: 2026-03-09
-- Purpose: Replace 5 DynamoDB tables with Supabase PostgreSQL tables
-- =============================================================================

-- Note: This migration runs against the BACKEND database (RECIPROCITY_BACKEND_DB_URL)
-- where the users table exists. The AI database (DATABASE_URL) only has embeddings.

-- =============================================================================
-- 1. user_profiles: Stores persona, Q&A data, and processing status
--    Replaces DynamoDB staging-user-profiles table
-- =============================================================================

CREATE TABLE IF NOT EXISTS user_profiles (
    -- Primary key matches users.id
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,

    -- Profile data (was ProfileData MapAttribute)
    resume_link TEXT,
    raw_questions JSONB DEFAULT '[]'::jsonb,  -- Array of {prompt, answer} objects

    -- Resume text (was ResumeTextData MapAttribute)
    resume_text TEXT,
    resume_extracted_at TIMESTAMPTZ,
    resume_extraction_method VARCHAR(50),

    -- Persona data (was PersonaData MapAttribute - 13 fields)
    persona_name VARCHAR(255),
    persona_archetype VARCHAR(100),
    persona_designation VARCHAR(255),
    persona_experience TEXT,
    persona_focus TEXT,
    persona_profile_essence TEXT,
    persona_strategy TEXT,  -- Replaces investment_philosophy
    persona_what_looking_for TEXT,
    persona_engagement_style TEXT,
    persona_requirements TEXT,
    persona_offerings TEXT,
    persona_user_type VARCHAR(100),
    persona_industry VARCHAR(100),
    persona_generated_at TIMESTAMPTZ,

    -- Processing status (was top-level attributes)
    processing_status VARCHAR(50) DEFAULT 'not_initiated',
    persona_status VARCHAR(50) DEFAULT 'not_initiated',
    needs_matchmaking BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_user_profiles_persona_status
    ON user_profiles(persona_status);
CREATE INDEX IF NOT EXISTS idx_user_profiles_needs_matchmaking
    ON user_profiles(needs_matchmaking) WHERE needs_matchmaking = TRUE;
CREATE INDEX IF NOT EXISTS idx_user_profiles_processing_status
    ON user_profiles(processing_status);

-- Auto-update trigger for updated_at
CREATE OR REPLACE FUNCTION update_user_profiles_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_user_profiles_updated_at ON user_profiles;
CREATE TRIGGER trigger_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_user_profiles_updated_at();

-- =============================================================================
-- 2. user_match_cache: Caches calculated matches for quick retrieval
--    Replaces DynamoDB user_matches table
--    NOTE: The backend 'matches' table is the source of truth for actual matches
--    This is a cache for AI service match calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS user_match_cache (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,

    -- Match data stored as JSONB for flexibility
    -- Contains: {requirements_matches: [...], offerings_matches: [...]}
    matches JSONB DEFAULT '{}'::jsonb,

    -- Metadata
    total_matches INTEGER DEFAULT 0,
    algorithm VARCHAR(50),  -- 'basic', 'enhanced', 'multi_vector', 'hybrid'
    threshold DECIMAL(4,3),

    -- Timestamps
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_match_cache_total_matches
    ON user_match_cache(total_matches) WHERE total_matches > 0;
CREATE INDEX IF NOT EXISTS idx_user_match_cache_last_updated
    ON user_match_cache(last_updated);

-- =============================================================================
-- 3. notified_match_pairs: Tracks which user pairs have been notified
--    Replaces DynamoDB notified_match_pairs table
--    Uses composite key pattern: sorted(user_a, user_b) ensures A-B == B-A
-- =============================================================================

CREATE TABLE IF NOT EXISTS notified_match_pairs (
    -- Composite primary key: always store in sorted order
    user_a_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_b_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Notification tracking
    notified_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    notification_count INTEGER DEFAULT 1,
    last_similarity_score DECIMAL(5,4),

    -- Ensure user_a_id < user_b_id for consistent pair representation
    CONSTRAINT pk_notified_match_pairs PRIMARY KEY (user_a_id, user_b_id),
    CONSTRAINT chk_user_ordering CHECK (user_a_id < user_b_id)
);

-- Indexes for querying by user
CREATE INDEX IF NOT EXISTS idx_notified_pairs_user_a
    ON notified_match_pairs(user_a_id);
CREATE INDEX IF NOT EXISTS idx_notified_pairs_user_b
    ON notified_match_pairs(user_b_id);
CREATE INDEX IF NOT EXISTS idx_notified_pairs_notified_at
    ON notified_match_pairs(notified_at);

-- =============================================================================
-- Notes on existing tables (NO CHANGES NEEDED):
--
-- Feedback → Use existing 'match_feedback' table in backend
-- ChatRecord → Use existing 'ai_conversations' table in backend
-- =============================================================================

-- =============================================================================
-- Helper function: Get or create user profile
-- =============================================================================

CREATE OR REPLACE FUNCTION get_or_create_user_profile(p_user_id UUID)
RETURNS user_profiles AS $$
DECLARE
    profile user_profiles;
BEGIN
    SELECT * INTO profile FROM user_profiles WHERE user_id = p_user_id;

    IF NOT FOUND THEN
        INSERT INTO user_profiles (user_id)
        VALUES (p_user_id)
        RETURNING * INTO profile;
    END IF;

    RETURN profile;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Helper function: Mark notification pair (handles ordering automatically)
-- =============================================================================

CREATE OR REPLACE FUNCTION mark_pair_notified(
    p_user_id_1 UUID,
    p_user_id_2 UUID,
    p_similarity_score DECIMAL(5,4) DEFAULT NULL
)
RETURNS void AS $$
DECLARE
    v_user_a UUID;
    v_user_b UUID;
BEGIN
    -- Ensure consistent ordering (a < b)
    IF p_user_id_1 < p_user_id_2 THEN
        v_user_a := p_user_id_1;
        v_user_b := p_user_id_2;
    ELSE
        v_user_a := p_user_id_2;
        v_user_b := p_user_id_1;
    END IF;

    INSERT INTO notified_match_pairs (user_a_id, user_b_id, last_similarity_score)
    VALUES (v_user_a, v_user_b, p_similarity_score)
    ON CONFLICT (user_a_id, user_b_id) DO UPDATE SET
        notification_count = notified_match_pairs.notification_count + 1,
        notified_at = CURRENT_TIMESTAMP,
        last_similarity_score = COALESCE(EXCLUDED.last_similarity_score, notified_match_pairs.last_similarity_score);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Helper function: Check if pair is notified
-- =============================================================================

CREATE OR REPLACE FUNCTION is_pair_notified(p_user_id_1 UUID, p_user_id_2 UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_user_a UUID;
    v_user_b UUID;
BEGIN
    -- Ensure consistent ordering
    IF p_user_id_1 < p_user_id_2 THEN
        v_user_a := p_user_id_1;
        v_user_b := p_user_id_2;
    ELSE
        v_user_a := p_user_id_2;
        v_user_b := p_user_id_1;
    END IF;

    RETURN EXISTS (
        SELECT 1 FROM notified_match_pairs
        WHERE user_a_id = v_user_a AND user_b_id = v_user_b
    );
END;
$$ LANGUAGE plpgsql;
