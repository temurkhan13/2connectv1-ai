"""create pgvector extension and user_embeddings table

Revision ID: ba55906aff3a
Revises: 
Create Date: 2025-09-28 05:31:59.421417

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ba55906aff3a'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create user_embeddings table
    op.create_table(
        'user_embeddings',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('embedding_type', sa.String(50), nullable=False),
        sa.Column('vector_data', sa.Text, nullable=True),  # Will store as vector type
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP AT TIME ZONE \'UTC\')')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP AT TIME ZONE \'UTC\')'))
    )
    
    # Create indexes
    op.create_index('user_embeddings_user_id_idx', 'user_embeddings', ['user_id'])
    op.create_index('user_embeddings_type_idx', 'user_embeddings', ['embedding_type'])
    
    # Create unique constraint
    op.create_index('user_embeddings_unique_idx', 'user_embeddings', ['user_id', 'embedding_type'], unique=True)
    
    # Convert text column to vector type (pgvector specific)
    op.execute("ALTER TABLE user_embeddings ALTER COLUMN vector_data TYPE vector(1536) USING vector_data::vector")
    
    # Create vector index for similarity search
    op.execute("CREATE INDEX user_embeddings_vector_idx ON user_embeddings USING ivfflat (vector_data vector_cosine_ops) WITH (lists = 100)")
    
    # Create similarity search function
    op.execute("""
        CREATE OR REPLACE FUNCTION find_similar_users(
            query_vector vector(1536),
            embedding_type_filter VARCHAR(50) DEFAULT NULL,
            limit_count INTEGER DEFAULT 10,
            similarity_threshold FLOAT DEFAULT 0.7
        )
        RETURNS TABLE(
            user_id VARCHAR(255),
            similarity_score FLOAT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                ue.user_id,
                1 - (ue.vector_data <=> query_vector) AS similarity
            FROM user_embeddings ue
            WHERE 
                (embedding_type_filter IS NULL OR ue.embedding_type = embedding_type_filter)
                AND (1 - (ue.vector_data <=> query_vector)) >= similarity_threshold
            ORDER BY ue.vector_data <=> query_vector
            LIMIT limit_count;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    # Drop function
    op.execute("DROP FUNCTION IF EXISTS find_similar_users")
    
    # Drop indexes
    op.drop_index('user_embeddings_vector_idx')
    op.drop_index('user_embeddings_unique_idx')
    op.drop_index('user_embeddings_type_idx')
    op.drop_index('user_embeddings_user_id_idx')
    
    # Drop table
    op.drop_table('user_embeddings')
    
    # Drop extension (careful - might affect other tables)
    # op.execute("DROP EXTENSION IF EXISTS vector")
