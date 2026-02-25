"""add_metadata_column_to_embeddings

Revision ID: 7a89aea925fc
Revises: ba55906aff3a
Create Date: 2025-09-29 01:05:05.739492

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7a89aea925fc'
down_revision: Union[str, None] = 'ba55906aff3a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add metadata column to user_embeddings table
    op.add_column('user_embeddings', sa.Column('metadata', sa.JSON, nullable=True))


def downgrade() -> None:
    # Remove metadata column
    op.drop_column('user_embeddings', 'metadata')
