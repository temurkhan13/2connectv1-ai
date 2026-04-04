-- Migration: Add ideal match profile columns for criteria-based matching
-- Date: 2026-04-04
--
-- Stores the LLM-generated description of each user's ideal match.
-- This description is embedded and used for cosine search against
-- other users' profile embeddings.

ALTER TABLE user_profiles
ADD COLUMN IF NOT EXISTS ideal_match_profile TEXT DEFAULT '',
ADD COLUMN IF NOT EXISTS ideal_match_hash VARCHAR(64) DEFAULT '';

-- Verify
SELECT 'Added ideal_match_profile + ideal_match_hash columns' as status;
