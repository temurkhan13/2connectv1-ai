from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20251124_resize_pgvector_column'
down_revision = '36e01bb2a501'
branch_labels = None
depends_on = None

def upgrade():
    # Drop and recreate the vector column with desired dimension.
    import os
    new_dim = int(os.getenv('EMBEDDING_DIMENSION', '768'))
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # WARNING: This drops existing vectors; run regeneration after this migration.
    op.execute("ALTER TABLE user_embeddings DROP COLUMN IF EXISTS vector_data")
    op.execute(f"ALTER TABLE user_embeddings ADD COLUMN vector_data vector({new_dim})")

def downgrade():
    # Recreate previous default dimension
    op.execute("ALTER TABLE user_embeddings DROP COLUMN IF EXISTS vector_data")
    op.execute("ALTER TABLE user_embeddings ADD COLUMN vector_data vector(1536)")
