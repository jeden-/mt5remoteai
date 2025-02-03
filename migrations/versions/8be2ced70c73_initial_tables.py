"""initial_tables

Revision ID: 8be2ced70c73
Revises: 
Create Date: 2025-02-01 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8be2ced70c73'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Tabela dla danych rynkowych
    op.create_table('market_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('open', sa.Numeric(), nullable=False),
        sa.Column('high', sa.Numeric(), nullable=False),
        sa.Column('low', sa.Numeric(), nullable=False),
        sa.Column('close', sa.Numeric(), nullable=False),
        sa.Column('volume', sa.Numeric(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timestamp')
    )
    
    op.create_index('idx_market_data_lookup', 'market_data',
                    ['symbol', 'timestamp'])

    # Tabela dla transakcji
    op.create_table('trades',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('order_type', sa.String(length=10), nullable=False),
        sa.Column('volume', sa.Numeric(), nullable=False),
        sa.Column('price', sa.Numeric(), nullable=False),
        sa.Column('sl', sa.Numeric(), nullable=True),
        sa.Column('tp', sa.Numeric(), nullable=True),
        sa.Column('open_time', sa.DateTime(), nullable=False),
        sa.Column('close_time', sa.DateTime(), nullable=True),
        sa.Column('profit', sa.Numeric(), nullable=True),
        sa.Column('status', sa.String(length=10), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'open_time', 'order_type')
    )
    
    op.create_index('idx_trades_lookup', 'trades',
                    ['symbol', 'open_time', 'status'])

    # Tabela dla danych historycznych
    op.create_table('historical_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('open', sa.Numeric(), nullable=False),
        sa.Column('high', sa.Numeric(), nullable=False),
        sa.Column('low', sa.Numeric(), nullable=False),
        sa.Column('close', sa.Numeric(), nullable=False),
        sa.Column('volume', sa.Numeric(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol', 'timeframe', 'timestamp')
    )
    
    op.create_index('idx_historical_data_lookup', 'historical_data',
                    ['symbol', 'timeframe', 'timestamp'])


def downgrade() -> None:
    op.drop_index('idx_historical_data_lookup', table_name='historical_data')
    op.drop_table('historical_data')
    op.drop_index('idx_trades_lookup', table_name='trades')
    op.drop_table('trades')
    op.drop_index('idx_market_data_lookup', table_name='market_data')
    op.drop_table('market_data') 