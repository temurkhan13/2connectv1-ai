"""update_vector_dimension_to_1536

Revision ID: 36e01bb2a501
Revises: 64db0b3be4
Create Date: 2025-11-17 15:15:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '36e01bb2a501'
down_revision: Union[str, None] = '64db0b3be4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update vector dimension from 1024 to 1536 for OpenAI text-embedding-3-small
    op.execute("ALTER TABLE user_embeddings ALTER COLUMN vector_data TYPE vector(1536)")
    
    # Update the similarity search function to use 1536 dimensions
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
    # Revert vector dimension back to 1024
    op.execute("ALTER TABLE user_embeddings ALTER COLUMN vector_data TYPE vector(1024)")
    
    # Revert the similarity search function
    op.execute("""
        CREATE OR REPLACE FUNCTION find_similar_users(
            query_vector vector(1024),
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