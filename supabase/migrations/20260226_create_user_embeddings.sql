-- Create user_embeddings table for Supabase (PostgreSQL with pgvector)
-- Run this in the Supabase SQL Editor: https://supabase.com/dashboard/project/{project_id}/sql

-- 1. Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create user_embeddings table
CREATE TABLE IF NOT EXISTS user_embeddings (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    embedding_type TEXT NOT NULL,
    vector_data VECTOR(1536),  -- 1536 dimensions for OpenAI text-embedding-3-small
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    updated_at TIMESTAMPTZ DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),

    -- Unique constraint for upsert operations
    UNIQUE(user_id, embedding_type)
);

-- 3. Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_embeddings_user_id ON user_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_user_embeddings_type ON user_embeddings(embedding_type);

-- 4. Create HNSW index for fast vector similarity search
-- This dramatically improves cosine similarity query performance
CREATE INDEX IF NOT EXISTS idx_user_embeddings_vector ON user_embeddings
USING hnsw (vector_data vector_cosine_ops);

-- 5. Grant permissions (adjust role if needed)
-- GRANT ALL ON user_embeddings TO authenticated;
-- GRANT ALL ON user_embeddings TO anon;

-- Verify table was created
SELECT 'user_embeddings table created successfully!' as status;
SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'user_embeddings';
