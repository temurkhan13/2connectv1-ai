-- Migration: Upgrade embeddings to gemini-embedding-2-preview @ 1536 dims
-- Model: gemini-embedding-2-preview (replaces deprecated text-embedding-004)
-- Date: 2026-04-04
--
-- Why 1536 not 3072: pgvector HNSW index has 2000 dimension limit.
-- Research shows 1536 = same MTEB quality as 3072 (68.17 score).
--
-- This migration:
-- 1. Drops existing HNSW index
-- 2. Clears old embeddings (incompatible model/dimensions)
-- 3. Keeps vector column at VECTOR(1536) (same size, new model)
-- 4. Recreates HNSW index
-- 5. Adds conversation_text column to user_profiles
--
-- IMPORTANT: All existing embeddings must be regenerated after this migration.

-- Step 1: Drop existing HNSW index
DROP INDEX IF EXISTS idx_user_embeddings_vector;

-- Step 2: Clear existing embeddings (old model, incompatible)
TRUNCATE user_embeddings;

-- Step 3: Ensure vector column is VECTOR(1536)
ALTER TABLE user_embeddings
ALTER COLUMN vector_data TYPE VECTOR(1536);

-- Step 4: Recreate HNSW index
CREATE INDEX IF NOT EXISTS idx_user_embeddings_vector ON user_embeddings
USING hnsw (vector_data vector_cosine_ops);

-- Step 5: Add conversation_text column to user_profiles
-- Stores the full onboarding conversation for richer AI summary generation
ALTER TABLE user_profiles
ADD COLUMN IF NOT EXISTS conversation_text TEXT DEFAULT '';

-- Verify
SELECT 'Migration complete: embeddings upgraded to gemini-embedding-2-preview @ 1536 dims' as status;
