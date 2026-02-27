-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create user_embeddings table
CREATE TABLE IF NOT EXISTS user_embeddings (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    embedding_type TEXT NOT NULL,
    vector_data VECTOR(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, embedding_type)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_user_embeddings_user_id ON user_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_user_embeddings_type ON user_embeddings(embedding_type);

-- HNSW index for fast vector similarity search
CREATE INDEX IF NOT EXISTS idx_user_embeddings_vector ON user_embeddings
USING hnsw (vector_data vector_cosine_ops);
