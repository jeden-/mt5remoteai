"""
Moduł zawierający wspólne fixtures dla testów.
"""
import pytest
import pytest_asyncio
import os
import asyncio
import asyncpg
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator, Dict, Any
from src.utils.config import ConfigLoader
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock
from pathlib import Path
import sys
from decimal import Decimal

from src.utils.logger import TradingLogger
from src.database.postgres_handler import PostgresHandler
from src.models.data_models import SignalData, SignalAction

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()


@pytest.fixture(scope="session")
def config() -> Dict[str, Any]:
    """Fixture dostarczający konfigurację testową."""
    return {
        "trading": {
            "symbol": "EURUSD",
            "timeframe": "1H",
            "max_position_size": 1.0,
            "stop_loss_pips": 50,
            "take_profit_pips": 100
        },
        "strategy": {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "sma_fast": 20,
            "sma_slow": 50
        },
        "database": {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "name": os.getenv("DB_NAME", "mt5remotetest"),
            "user": os.getenv("DB_USER", "mt5remote"),
            "password": os.getenv("DB_PASSWORD", "mt5remote"),
            "pool_min_size": int(os.getenv("DB_POOL_MIN_SIZE", "5")),
            "pool_max_size": int(os.getenv("DB_POOL_MAX_SIZE", "20")),
            "command_timeout": int(os.getenv("DB_COMMAND_TIMEOUT", "60")),
            "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
            "max_retries": int(os.getenv("DB_MAX_RETRIES", "3")),
            "retry_interval": int(os.getenv("DB_RETRY_INTERVAL", "5"))
        },
        "logging": {
            "level": "DEBUG",
            "file": "test.log",
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        }
    }


@pytest_asyncio.fixture(scope="session")
async def check_db_connection(config: Dict[str, Any]) -> bool:
    """Fixture sprawdzający dostępność bazy danych."""
    try:
        conn = await asyncpg.connect(
            host=config["database"]["host"],
            port=config["database"]["port"],
            user=config["database"]["user"],
            password=config["database"]["password"],
            database='postgres'
        )
        await conn.close()
        return True
    except Exception as e:
        pytest.skip(f"Baza danych jest niedostępna: {str(e)}")
        return False


@pytest_asyncio.fixture(scope="session")
async def create_test_db(config: Dict[str, Any], check_db_connection: bool) -> AsyncGenerator[None, None]:
    """Fixture tworzący bazę danych testową."""
    if not check_db_connection:
        pytest.skip("Baza danych jest niedostępna")
        return

    sys_conn = await asyncpg.connect(
        host=config["database"]["host"],
        port=config["database"]["port"],
        user=config["database"]["user"],
        password=config["database"]["password"],
        database='postgres'
    )
    
    try:
        await sys_conn.execute(f'DROP DATABASE IF EXISTS {config["database"]["name"]}')
        await sys_conn.execute(f'CREATE DATABASE {config["database"]["name"]}')
    finally:
        await sys_conn.close()

    # Połącz się z nową bazą i utwórz tabele
    conn = await asyncpg.connect(
        host=config["database"]["host"],
        port=config["database"]["port"],
        user=config["database"]["user"],
        password=config["database"]["password"],
        database=config["database"]["name"]
    )
    
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                UNIQUE(symbol, timestamp)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                order_type VARCHAR(4) NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                sl DOUBLE PRECISION,
                tp DOUBLE PRECISION,
                open_time TIMESTAMP NOT NULL,
                close_time TIMESTAMP,
                status VARCHAR(10) NOT NULL,
                profit DOUBLE PRECISION
            );

            CREATE TABLE IF NOT EXISTS historical_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                timeframe VARCHAR(3) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            );
        """)
    finally:
        await conn.close()

    yield

    # Usuń bazę po zakończeniu testów
    sys_conn = await asyncpg.connect(
        host=config["database"]["host"],
        port=config["database"]["port"],
        user=config["database"]["user"],
        password=config["database"]["password"],
        database='postgres'
    )
    
    try:
        await sys_conn.execute(f'DROP DATABASE IF EXISTS {config["database"]["name"]}')
    finally:
        await sys_conn.close()


@pytest_asyncio.fixture(scope="session")
async def test_db_pool(config: Dict[str, Any], create_test_db: None) -> AsyncGenerator[asyncpg.Pool, None]:
    """Fixture dostarczający pulę połączeń do testowej bazy danych."""
    pool = await asyncpg.create_pool(
        host=config["database"]["host"],
        port=config["database"]["port"],
        user=config["database"]["user"],
        password=config["database"]["password"],
        database=config["database"]["name"],
        min_size=config["database"]["pool_min_size"],
        max_size=config["database"]["pool_max_size"],
        command_timeout=config["database"]["command_timeout"]
    )
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def test_db_handler():
    """
    Fixture dostarczający handler bazy danych do testów.
    
    Returns:
        PostgresHandler: Handler bazy danych
    """
    handler = PostgresHandler(
        user='mt5remote',
        password='mt5remote',
        database='mt5remotetest',
        host='localhost',
        port=5432
    )
    await handler.initialize()
    yield handler
    await handler.close()


@pytest.fixture
def sample_market_data() -> dict:
    """
    Fixture dostarczający przykładowe dane rynkowe.
    
    Returns:
        dict: Słownik z danymi rynkowymi
    """
    return {
        'symbol': 'EURUSD',
        'current_price': 1.1000,
        'sma_20': 1.0990,
        'sma_50': 1.0980,
        'price_change_24h': 0.5,
        'volume_24h': 10000,
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_trade_info() -> dict:
    """
    Fixture dostarczający przykładowe dane transakcji.
    
    Returns:
        dict: Słownik z danymi transakcji
    """
    return {
        'symbol': 'EURUSD',
        'type': 'BUY',
        'volume': 0.1,
        'price': 1.1000,
        'sl': 1.0950,
        'tp': 1.1100,
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_ai_analysis() -> dict:
    """
    Fixture dostarczający przykładową analizę AI.
    
    Returns:
        dict: Słownik z analizą AI
    """
    return {
        'symbol': 'EURUSD',
        'timestamp': datetime.now().isoformat(),
        'ollama_analysis': {
            'trend': 'UP',
            'strength': 8,
            'recommendation': 'BUY'
        },
        'claude_analysis': {
            'recommendation': 'LONG',
            'confidence': 0.85,
            'risk_level': 'MEDIUM'
        }
    }


@pytest.fixture
def sample_error_info() -> dict:
    """
    Fixture dostarczający przykładowe informacje o błędzie.
    
    Returns:
        dict: Słownik z informacjami o błędzie
    """
    return {
        'type': 'CONNECTION_ERROR',
        'message': 'Nie można połączyć z MT5',
        'timestamp': datetime.now().isoformat(),
        'details': {
            'attempt': 3,
            'last_error': 'Connection timeout'
        }
    }


@pytest.fixture
def sample_strategy_config() -> dict:
    """
    Fixture dostarczający przykładową konfigurację strategii.
    
    Returns:
        dict: Słownik z konfiguracją strategii
    """
    return {
        'max_position_size': 0.1,
        'max_risk_per_trade': 0.02,
        'allowed_symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'timeframe': 'H1',
        'indicators': {
            'sma_periods': [20, 50],
            'rsi_period': 14
        }
    }


@pytest.fixture
def sample_account_info() -> dict:
    """
    Fixture dostarczający przykładowe informacje o koncie.
    
    Returns:
        dict: Słownik z informacjami o koncie
    """
    return {
        'balance': 10000,
        'equity': 10000,
        'margin': 0,
        'margin_level': 0,
        'profit': 0,
        'currency': 'USD',
        'leverage': 100
    }


@pytest_asyncio.fixture(scope="function")
async def clean_test_db(test_db_handler):
    """Fixture czyszczący bazę testową przed i po testach."""
    async with test_db_handler.pool.acquire() as conn:
        await conn.execute("""
            DROP TABLE IF EXISTS market_data CASCADE;
            DROP TABLE IF EXISTS trades CASCADE;
            DROP TABLE IF EXISTS historical_data CASCADE;
        """)
    await test_db_handler.create_tables()
    yield
    async with test_db_handler.pool.acquire() as conn:
        await conn.execute("""
            DROP TABLE IF EXISTS market_data CASCADE;
            DROP TABLE IF EXISTS trades CASCADE;
            DROP TABLE IF EXISTS historical_data CASCADE;
        """)


@pytest_asyncio.fixture
async def sample_db_data(test_db_pool: asyncpg.Pool) -> AsyncGenerator[None, None]:
    """
    Fixture wstawiający przykładowe dane do bazy testowej.
    
    Args:
        test_db_pool: Pula połączeń do bazy testowej
    """
    async with test_db_pool.acquire() as conn:
        # Przykładowe dane rynkowe
        await conn.execute("""
            INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
            VALUES 
                ('EURUSD', NOW(), 1.1000, 1.1100, 1.0900, 1.1050, 1000),
                ('GBPUSD', NOW(), 1.2500, 1.2600, 1.2400, 1.2550, 800);
        """)
        
        # Przykładowe transakcje
        await conn.execute("""
            INSERT INTO trades (symbol, order_type, volume, price, sl, tp, open_time, status)
            VALUES 
                ('EURUSD', 'BUY', 0.1, 1.1000, 1.0950, 1.1100, NOW(), 'OPEN'),
                ('GBPUSD', 'SELL', 0.1, 1.2550, 1.2600, 1.2450, NOW(), 'OPEN');
        """)
        
        # Przykładowe dane historyczne
        await conn.execute("""
            INSERT INTO historical_data (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES 
                ('EURUSD', '1H', NOW(), 1.1000, 1.1100, 1.0900, 1.1050, 1000),
                ('GBPUSD', '1H', NOW(), 1.2500, 1.2600, 1.2400, 1.2550, 800);
        """)
    
    yield 


@pytest.fixture
def sample_data():
    """Przykładowe dane historyczne dla testów."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame(index=dates)
    data['open'] = np.random.uniform(100, 110, len(dates))
    data['high'] = data['open'] + np.random.uniform(0, 2, len(dates))
    data['low'] = data['open'] - np.random.uniform(0, 2, len(dates))
    data['close'] = np.random.uniform(100, 110, len(dates))
    data['volume'] = np.random.uniform(1000, 5000, len(dates))
    
    # Dodaj wskaźniki techniczne
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['RSI'] = 50 + np.random.uniform(-20, 20, len(dates))
    data['MACD'] = np.random.uniform(-2, 2, len(dates))
    data['Signal_Line'] = np.random.uniform(-2, 2, len(dates))
    data['BB_upper'] = data['SMA_20'] + 2 * data['close'].rolling(window=20).std()
    data['BB_middle'] = data['SMA_20']
    data['BB_lower'] = data['SMA_20'] - 2 * data['close'].rolling(window=20).std()
    
    return data


@pytest.fixture
def small_sample_data():
    """Przykładowe dane historyczne dla testów - mały zestaw."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='1h')
    data = pd.DataFrame(index=dates)
    data['open'] = np.random.uniform(100, 110, len(dates))
    data['high'] = data['open'] + np.random.uniform(0, 2, len(dates))
    data['low'] = data['open'] - np.random.uniform(0, 2, len(dates))
    data['close'] = np.random.uniform(100, 110, len(dates))
    data['volume'] = np.random.uniform(1000, 5000, len(dates))
    
    # Dodaj wskaźniki techniczne
    data['SMA_20'] = data['close'].rolling(window=2).mean()
    data['SMA_50'] = data['close'].rolling(window=2).mean()
    data['RSI'] = 50 + np.random.uniform(-20, 20, len(dates))
    data['MACD'] = np.random.uniform(-2, 2, len(dates))
    data['Signal_Line'] = np.random.uniform(-2, 2, len(dates))
    
    return data


@pytest.fixture
def mock_logger():
    """Fixture tworzący mock dla loggera."""
    logger = Mock()
    logger.info = AsyncMock()
    logger.error = AsyncMock()
    logger.warning = AsyncMock()
    logger.debug = AsyncMock()
    logger.log_trade = AsyncMock()
    logger.log_error = AsyncMock()
    return logger


@pytest.fixture
def mock_db_handler():
    """Mock dla handlera bazy danych."""
    handler = Mock(spec=PostgresHandler)
    handler.fetch_all = AsyncMock()
    handler.execute_many = AsyncMock()
    return handler


@pytest.fixture
def sample_trades():
    """Przykładowe transakcje do testów."""
    return [
        {
            'entry_time': datetime(2024, 1, 1, 10, 0),
            'exit_time': datetime(2024, 1, 1, 11, 0),
            'entry_price': 1.1000,
            'exit_price': 1.1050,
            'direction': 'BUY',
            'profit': 50.0,
            'size': 1.0,
            'pips': 50
        },
        {
            'entry_time': datetime(2024, 1, 2, 10, 0),
            'exit_time': datetime(2024, 1, 2, 11, 0),
            'entry_price': 1.1100,
            'exit_price': 1.1000,
            'direction': 'SELL',
            'profit': -100.0,
            'size': 1.0,
            'pips': -100
        },
        {
            'entry_time': datetime(2024, 1, 3, 10, 0),
            'exit_time': datetime(2024, 1, 3, 11, 0),
            'entry_price': 1.1000,
            'exit_price': 1.1080,
            'direction': 'BUY',
            'profit': 80.0,
            'size': 1.0,
            'pips': 80
        }
    ]


@pytest.fixture
def temp_logs_dir(tmp_path):
    """Fixture tworzący tymczasowy katalog na logi."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    yield logs_dir
    
    # Czyszczenie po testach
    if logs_dir.exists():
        for file in logs_dir.glob("**/*"):
            try:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    file.rmdir()
            except Exception:
                pass
        logs_dir.rmdir()


@pytest.fixture(scope="session")
def event_loop_policy():
    """Fixture ustawiający politykę event loop."""
    if sys.platform == 'win32':
        policy = asyncio.WindowsSelectorEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)
    return asyncio.get_event_loop_policy()


@pytest.fixture(scope="function")
def event_loop():
    """Fixture tworzący event loop dla testów asynchronicznych."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def position_manager_fixture(mock_logger):
    """
    Fixture dla PositionManager z własnym event loop.
    """
    from src.trading.position_manager import PositionManager
    from decimal import Decimal
    
    manager = PositionManager(
        symbol='EURUSD',
        max_position_size=Decimal('1.0'),
        stop_loss_pips=Decimal('50'),
        take_profit_pips=Decimal('100'),
        logger=mock_logger
    )
    yield manager


@pytest.fixture
def sample_signal():
    """Fixture tworzący przykładowy sygnał BUY."""
    return SignalData(
        timestamp=datetime(2025, 2, 3, 9, 14, 30, 74923),
        symbol='EURUSD',
        action=SignalAction.BUY,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.0950'),
        take_profit=Decimal('1.1100'),
        indicators={},
        ai_analysis={}
    )


@pytest.fixture
def sample_sell_signal():
    """Fixture tworzący przykładowy sygnał SELL."""
    return SignalData(
        timestamp=datetime(2025, 2, 3, 9, 14, 30, 574926),
        symbol='EURUSD',
        action=SignalAction.SELL,
        confidence=0.95,
        entry_price=Decimal('1.1000'),
        price=Decimal('1.1000'),
        volume=Decimal('0.1'),
        stop_loss=Decimal('1.1050'),
        take_profit=Decimal('1.0900'),
        indicators={},
        ai_analysis={}
    ) 