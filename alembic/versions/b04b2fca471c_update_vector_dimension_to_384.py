"""update_vector_dimension_to_384

Revision ID: b04b2fca471c
Revises: 7a89aea925fc
Create Date: 2025-09-29 02:19:25.555358

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b04b2fca471c'
down_revision: Union[str, None] = '7a89aea925fc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update vector dimension from 1536 to 384 for SentenceTransformers
    op.execute("ALTER TABLE user_embeddings ALTER COLUMN vector_data TYPE vector(384)")
    
    # Update the similarity search function to use 384 dimensions
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


def downgrade() -> None:
    # Revert vector dimension back to 1536
    op.execute("ALTER TABLE user_embeddings ALTER COLUMN vector_data TYPE vector(1536)")
    
    # Revert the similarity search function
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
