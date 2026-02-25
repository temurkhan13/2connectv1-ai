"""update_vector_dimension_to_1024

Revision ID: 64db0b3be4
Revises: b04b2fca471c
Create Date: 2025-11-17 15:06:46

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '64db0b3be4'
down_revision: Union[str, None] = 'b04b2fca471c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update vector dimension from 384 to 1024 for larger embeddings
    op.execute("ALTER TABLE user_embeddings ALTER COLUMN vector_data TYPE vector(1024)")
    
    # Update the similarity search function to use 1024 dimensions
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


def downgrade() -> None:
    # Revert vector dimension back to 384
    op.execute("ALTER TABLE user_embeddings ALTER COLUMN vector_data TYPE vector(384)")
    
    # Revert the similarity search function
    op.execute("""
        CREATE OR REPLACE FUNCTION find_similar_users(
            query_vector vector(384),
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