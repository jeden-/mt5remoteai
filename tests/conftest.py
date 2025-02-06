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
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path
import sys
from decimal import Decimal

from src.utils.logger import TradingLogger
from src.database.postgres_handler import PostgresHandler
from src.models.data_models import SignalData, SignalAction

# Załaduj zmienne środowiskowe z pliku .env
load_dotenv()


class AsyncMockWrapper:
    """Wrapper dla asynchronicznych mocków."""
    
    def __init__(self, mock_name: str):
        self._mock = AsyncMock()
        self._mock_name = mock_name
        self._calls = []
        
    async def __call__(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        return await self._mock(*args, **kwargs)
        
    @property
    def call_count(self) -> int:
        return len(self._calls)
        
    def assert_called_once(self):
        assert len(self._calls) == 1, f"Expected {self._mock_name} to be called once. Called {len(self._calls)} times."
        
    def assert_called_with(self, *args, **kwargs):
        assert (args, kwargs) in self._calls, f"Expected {self._mock_name} to be called with {args}, {kwargs}"
        
    def assert_not_called(self):
        assert len(self._calls) == 0, f"Expected {self._mock_name} not to be called. Called {len(self._calls)} times."
        
    def reset_mock(self):
        self._calls = []
        self._mock.reset_mock()


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
        'timestamp': datetime.now(),
        'open': 1.1000,
        'high': 1.1050,
        'low': 1.0950,
        'close': 1.1025,
        'volume': 1000.0
    }


@pytest.fixture
def sample_trade_info() -> dict:
    """
    Fixture dostarczający przykładowe dane o transakcji.
    
    Returns:
        dict: Słownik z danymi o transakcji
    """
    return {
        'symbol': 'EURUSD',
        'order_type': 'BUY',
        'volume': 0.1,
        'price': 1.1000,
        'sl': 1.0950,
        'tp': 1.1100,
        'open_time': datetime.now(),
        'status': 'OPEN'
    }


@pytest.fixture
def sample_ai_analysis() -> dict:
    """
    Fixture dostarczający przykładową analizę AI.
    
    Returns:
        dict: Słownik z analizą AI
    """
    return {
        'sentiment': 'BULLISH',
        'confidence': 0.85,
        'factors': [
            'Strong upward trend',
            'Positive market sentiment',
            'Low volatility'
        ],
        'risk_level': 'MEDIUM',
        'recommended_position_size': 0.1,
        'timestamp': datetime.now()
    }


@pytest.fixture
def sample_error_info() -> dict:
    """
    Fixture dostarczający przykładowe informacje o błędzie.
    
    Returns:
        dict: Słownik z informacjami o błędzie
    """
    return {
        'error_code': 1000,
        'error_message': 'Connection error',
        'timestamp': datetime.now(),
        'severity': 'HIGH',
        'component': 'MT5_CONNECTOR',
        'details': 'Failed to connect to MT5 terminal'
    }


@pytest.fixture
def sample_strategy_config() -> dict:
    """
    Fixture dostarczający przykładową konfigurację strategii.
    
    Returns:
        dict: Słownik z konfiguracją strategii
    """
    return {
        'name': 'RSI_Strategy',
        'timeframe': '1H',
        'parameters': {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'stop_loss_pips': 50,
            'take_profit_pips': 100
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
        'balance': 10000.0,
        'equity': 10100.0,
        'margin': 100.0,
        'free_margin': 9900.0,
        'margin_level': 1010.0,
        'leverage': 100,
        'currency': 'USD'
    }


@pytest_asyncio.fixture(scope="function")
async def clean_test_db(test_db_handler):
    """
    Fixture czyszczący bazę danych przed każdym testem.
    
    Args:
        test_db_handler: Handler bazy danych
    """
    async with test_db_handler.pool.acquire() as conn:
        await conn.execute("TRUNCATE market_data, trades, historical_data RESTART IDENTITY")
    yield


@pytest_asyncio.fixture
async def sample_db_data(test_db_pool: asyncpg.Pool) -> AsyncGenerator[None, None]:
    """
    Fixture wstawiający przykładowe dane do bazy.
    
    Args:
        test_db_pool: Pula połączeń do bazy danych
    """
    async with test_db_pool.acquire() as conn:
        # Wstaw przykładowe dane rynkowe
        await conn.execute("""
            INSERT INTO market_data (symbol, timestamp, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, 'EURUSD', datetime.now(), 1.1000, 1.1050, 1.0950, 1.1025, 1000.0)
        
        # Wstaw przykładową transakcję
        await conn.execute("""
            INSERT INTO trades (symbol, order_type, volume, price, sl, tp, open_time, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, 'EURUSD', 'BUY', 0.1, 1.1000, 1.0950, 1.1100, datetime.now(), 'OPEN')
    
    yield


@pytest.fixture
def sample_data():
    """Fixture dostarczający przykładowe dane do testów."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Generuj losowe dane OHLCV
    data['open'] = np.random.normal(1.1000, 0.0010, size=len(data))
    data['high'] = data['open'] + abs(np.random.normal(0, 0.0005, size=len(data)))
    data['low'] = data['open'] - abs(np.random.normal(0, 0.0005, size=len(data)))
    data['close'] = np.random.normal(1.1000, 0.0010, size=len(data))
    data['volume'] = np.random.normal(1000, 100, size=len(data))
    
    return data


@pytest.fixture
def small_sample_data():
    """Fixture dostarczający mały zestaw danych do testów."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
    data = pd.DataFrame(index=dates)
    
    # Generuj deterministyczne dane OHLCV
    data['open'] = [1.1000, 1.1010, 1.1020, 1.1015, 1.1025, 1.1035, 1.1030, 1.1040, 1.1045, 1.1050]
    data['high'] = [1.1020, 1.1030, 1.1040, 1.1035, 1.1045, 1.1055, 1.1050, 1.1060, 1.1065, 1.1070]
    data['low'] = [1.0990, 1.1000, 1.1010, 1.1005, 1.1015, 1.1025, 1.1020, 1.1030, 1.1035, 1.1040]
    data['close'] = [1.1010, 1.1020, 1.1015, 1.1025, 1.1035, 1.1030, 1.1040, 1.1045, 1.1050, 1.1060]
    data['volume'] = [1000, 1100, 900, 1200, 1000, 1300, 800, 1100, 1000, 1200]
    
    return data


@pytest_asyncio.fixture
async def mock_logger():
    """
    Tworzy mock obiektu loggera do testów.
    """
    logger = AsyncMock(spec=TradingLogger)
    
    # Dodaj atrybuty do śledzenia wywołań
    logger.info_calls = []
    logger.error_calls = []
    logger.warning_calls = []
    logger.debug_calls = []
    logger.log_trade_calls = []
    logger.log_error_calls = []
    
    # Zdefiniuj zachowanie metod
    async def info_side_effect(message):
        logger.info_calls.append(message)
        
    async def error_side_effect(message):
        logger.error_calls.append(message)
        
    async def warning_side_effect(message):
        logger.warning_calls.append(message)
        
    async def debug_side_effect(message):
        logger.debug_calls.append(message)
        
    async def log_trade_side_effect(data, action=None):
        logger.log_trade_calls.append((data, action))
        
    async def log_error_side_effect(error):
        logger.log_error_calls.append(error)
    
    # Przypisz side effects
    logger.info.side_effect = info_side_effect
    logger.error.side_effect = error_side_effect
    logger.warning.side_effect = warning_side_effect
    logger.debug.side_effect = debug_side_effect
    logger.log_trade.side_effect = log_trade_side_effect
    logger.log_error.side_effect = log_error_side_effect
    
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


@pytest_asyncio.fixture
async def position_manager(mock_logger):
    """Fixture zwracająca menedżera pozycji."""
    from src.trading.position_manager import PositionManager
    from decimal import Decimal
    
    manager = PositionManager(
        symbol='EURUSD',
        max_position_size=Decimal('1.0'),
        stop_loss_pips=Decimal('50'),
        take_profit_pips=Decimal('100'),
        trailing_stop_pips=Decimal('30'),
        logger=mock_logger
    )
    yield manager 