"""
Testy jednostkowe dla modułu mt5_handler.py
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import asyncio
import memory_profiler
from decimal import Decimal

from src.trading.mt5_handler import MT5Handler
from src.utils.logger import TradingLogger
from src.trading.trade_type import TradeType
from src.trading.position_status import PositionStatus

@pytest.fixture
def mock_logger():
    """Fixture dla mocka loggera."""
    logger = Mock(spec=TradingLogger)
    logger.error = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.debug = Mock()
    return logger

@pytest.fixture
def handler(mock_logger):
    """Fixture dla MT5Handler."""
    with patch('MetaTrader5.initialize') as mock_init, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info:
        mock_init.return_value = True
        mock_symbol_info.return_value = Mock()  # Zwraca mock zamiast None
        handler = MT5Handler(
            symbol='EURUSD',
            timeframe='1H',
            logger=mock_logger
        )
        return handler

def test_initialization():
    """Test inicjalizacji handlera MT5."""
    with patch('MetaTrader5.initialize') as mock_init, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info:
        mock_init.return_value = True
        mock_symbol_info.return_value = Mock()
        logger = Mock(spec=TradingLogger)
        logger.error = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.debug = Mock()

        handler = MT5Handler('EURUSD', logger=logger)
        
        assert handler.symbol == 'EURUSD'
        assert handler.timeframe == mt5.TIMEFRAME_H1
        assert handler.logger == logger
        mock_init.assert_called_once()

def test_initialization_error():
    """Test błędu podczas inicjalizacji MT5."""
    with patch('MetaTrader5.initialize') as mock_init:
        mock_init.return_value = False
        logger = Mock(spec=TradingLogger)
        logger.error = Mock()
        logger.info = Mock()
        logger.warning = Mock()
        logger.debug = Mock()

        with pytest.raises(RuntimeError) as exc_info:
            MT5Handler('EURUSD', logger=logger)
            
        assert str(exc_info.value) == "Nie udało się zainicjalizować MT5"
        mock_init.assert_called_once()

@pytest.mark.asyncio
async def test_open_position(handler):
    """Test otwierania pozycji."""
    with patch('MetaTrader5.order_send') as mock_send, \
         patch('MetaTrader5.symbol_info_tick') as mock_tick, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info:
        
        mock_tick.return_value = Mock(
            bid=1.1000,
            ask=1.1002
        )
        mock_info = Mock()
        mock_info.volume_min = 0.01
        mock_info.volume_max = 100.0
        mock_info.volume_step = 0.01
        mock_symbol_info.return_value = mock_info
        
        mock_send.return_value = Mock(
            retcode=mt5.TRADE_RETCODE_DONE,
            volume=1.0,
            price=1.1000,
            comment="Test trade"
        )

        result = await handler.open_position(
            direction='BUY',
            volume=1.0,
            stop_loss=1.0950,
            take_profit=1.1050
        )

        assert result['status'] == 'success'
        assert result['volume'] == 1.0
        assert result['price'] == 1.1000

@pytest.mark.asyncio
async def test_open_position_error(handler):
    """Test błędu podczas otwierania pozycji."""
    with patch('MetaTrader5.order_send') as mock_send, \
         patch('MetaTrader5.symbol_info_tick') as mock_tick, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info:
        
        mock_tick.return_value = Mock(
            bid=1.1000,
            ask=1.1002
        )
        mock_info = Mock()
        mock_info.volume_min = 0.01
        mock_info.volume_max = 100.0
        mock_info.volume_step = 0.01
        mock_symbol_info.return_value = mock_info
        
        mock_send.return_value = Mock(
            retcode=mt5.TRADE_RETCODE_ERROR,
            comment="Test error"
        )

        result = await handler.open_position(
            direction='BUY',
            volume=1.0
        )

        assert result['status'] == 'error'
        assert result['message'] == "Test error"
        assert result['code'] == mt5.TRADE_RETCODE_ERROR

@pytest.mark.asyncio
async def test_close_position(handler):
    """Test zamykania pozycji."""
    with patch('MetaTrader5.positions_get') as mock_get, \
         patch('MetaTrader5.order_send') as mock_send, \
         patch('MetaTrader5.symbol_info_tick') as mock_tick, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info:

        # Symuluj otwartą pozycję
        mock_position = Mock(
            ticket=12345,
            type=mt5.ORDER_TYPE_BUY,
            volume=1.0,
            price_open=1.1000,
            price_current=1.1050,
            profit=50.0
        )
        mock_get.return_value = [mock_position]

        mock_tick.return_value = Mock(
            bid=1.1050,
            ask=1.1052
        )
        mock_symbol_info.return_value = Mock(
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01
        )

        # Symuluj udane zamknięcie
        mock_send.return_value = Mock(
            retcode=mt5.TRADE_RETCODE_DONE,
            volume=1.0,
            price=1.1050,
            comment="Test close"
        )

        result = await handler.close_position()

        assert result['status'] == 'success'
        assert result['volume'] == 1.0
        assert result['price'] == 1.1050

@pytest.mark.asyncio
async def test_close_position_no_position(handler):
    """Test zamykania pozycji gdy nie ma otwartej."""
    with patch('MetaTrader5.positions_get') as mock_get:
        mock_get.return_value = []
        
        result = await handler.close_position()
        
        assert result['status'] == 'error'
        assert 'Brak otwartej pozycji' in result['message']

@pytest.mark.asyncio
async def test_get_current_price(handler):
    """Test pobierania aktualnej ceny."""
    with patch('MetaTrader5.symbol_info_tick') as mock_tick:
        mock_tick.return_value = Mock(
            bid=1.1000,
            ask=1.1002,
            last=1.1001,
            volume=1000,
            time=datetime.now().timestamp()
        )
        
        price = await handler.get_current_price()
        
        assert isinstance(price, dict)
        assert 'bid' in price
        assert 'ask' in price
        assert 'last' in price
        assert price['bid'] == 1.1000
        assert price['ask'] == 1.1002
        mock_tick.assert_called_once_with('EURUSD')

@pytest.mark.asyncio
async def test_get_historical_data(handler):
    """Test pobierania danych historycznych."""
    with patch('MetaTrader5.copy_rates_from') as mock_copy:
        # Przygotuj przykładowe dane
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        data = []
        for date in dates:
            data.append({
                'time': int(date.timestamp()),
                'open': np.random.uniform(1.1000, 1.1100),
                'high': np.random.uniform(1.1100, 1.1200),
                'low': np.random.uniform(1.0900, 1.1000),
                'close': np.random.uniform(1.1000, 1.1100),
                'tick_volume': np.random.randint(1000, 2000),
                'spread': 2,
                'real_volume': np.random.randint(10000, 20000)
            })
        mock_copy.return_value = data
        
        df = await handler.get_historical_data(
            start_date=datetime(2024, 1, 1),
            num_bars=100
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert all(col in df.columns for col in [
            'open', 'high', 'low', 'close', 'volume'
        ])
        mock_copy.assert_called_once()

@pytest.mark.asyncio
async def test_get_historical_data_error(handler):
    """Test błędu podczas pobierania danych historycznych."""
    with patch('MetaTrader5.copy_rates_from') as mock_copy:
        mock_copy.return_value = None
        
        with pytest.raises(RuntimeError) as exc_info:
            await handler.get_historical_data(
                start_date=datetime(2024, 1, 1),
                num_bars=100
            )
        
        assert "Nie udało się pobrać danych historycznych" in str(exc_info.value)

@pytest.mark.asyncio
async def test_get_account_info(handler):
    """Test pobierania informacji o koncie."""
    with patch('MetaTrader5.account_info') as mock_info:
        mock_info.return_value = Mock(
            login=12345,
            balance=10000.0,
            equity=10050.0,
            margin=100.0,
            margin_free=9950.0,
            margin_level=100.5,
            currency='USD'
        )
        
        info = await handler.get_account_info()
        
        assert isinstance(info, dict)
        assert info['balance'] == 10000.0
        assert info['equity'] == 10050.0
        assert info['margin'] == 100.0
        assert info['currency'] == 'USD'
        mock_info.assert_called_once()

@pytest.mark.asyncio
async def test_get_positions(handler):
    """Test pobierania otwartych pozycji."""
    with patch('MetaTrader5.positions_get') as mock_get:
        mock_positions = [
            Mock(
                ticket=12345,
                symbol='EURUSD',
                type=mt5.ORDER_TYPE_BUY,
                volume=1.0,
                price_open=1.1000,
                price_current=1.1050,
                profit=50.0,
                sl=1.0950,
                tp=1.1100,
                time=1234567890,  # Unix timestamp
                swap=0.0
            ),
            Mock(
                ticket=12346,
                symbol='GBPUSD',
                type=mt5.ORDER_TYPE_SELL,
                volume=0.5,
                price_open=1.2500,
                price_current=1.2450,
                profit=25.0,
                sl=1.2550,
                tp=1.2400,
                time=1234567890,  # Unix timestamp
                swap=0.0
            )
        ]
        mock_get.return_value = mock_positions
        
        positions = await handler.get_positions()
        
        assert len(positions) == 2
        assert positions[0]['ticket'] == 12345
        assert positions[0]['symbol'] == 'EURUSD'
        assert positions[0]['type'] == 'BUY'
        assert positions[0]['volume'] == 1.0
        assert positions[0]['profit'] == 50.0
        assert positions[1]['ticket'] == 12346
        assert positions[1]['symbol'] == 'GBPUSD'
        assert positions[1]['type'] == 'SELL'
        assert positions[1]['volume'] == 0.5
        assert positions[1]['profit'] == 25.0
        mock_get.assert_called_once()

def test_convert_timeframe():
    """Test konwersji timeframe na format MT5."""
    with patch('MetaTrader5.symbol_info') as mock_symbol_info:
        mock_symbol_info.return_value = Mock()

        handler = MT5Handler('EURUSD', timeframe='1H')
        assert handler.timeframe == mt5.TIMEFRAME_H1

        handler = MT5Handler('EURUSD', timeframe='4H')
        assert handler.timeframe == mt5.TIMEFRAME_H4

        handler = MT5Handler('EURUSD', timeframe='1D')
        assert handler.timeframe == mt5.TIMEFRAME_D1

        with pytest.raises(ValueError) as exc_info:
            MT5Handler('EURUSD', timeframe='INVALID')
        assert "Nieprawidłowy timeframe: INVALID" in str(exc_info.value)

def test_cleanup(handler):
    """Test czyszczenia zasobów."""
    with patch('MetaTrader5.shutdown') as mock_shutdown:
        handler.cleanup()
        mock_shutdown.assert_called_once()

def test_invalid_symbol():
    """Test inicjalizacji z nieprawidłowym symbolem."""
    with patch('MetaTrader5.initialize') as mock_init, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info:
        mock_init.return_value = True
        mock_symbol_info.return_value = None
        logger = Mock(spec=TradingLogger)
        
        with pytest.raises(ValueError, match="Nieprawidłowy symbol"):
            MT5Handler('INVALID', logger=logger)
        
        mock_symbol_info.assert_called_once_with('INVALID')

def test_invalid_timeframe():
    """Test inicjalizacji z nieprawidłowym timeframe."""
    with patch('MetaTrader5.initialize') as mock_init, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info:
        mock_init.return_value = True
        mock_symbol_info.return_value = Mock()
        logger = Mock(spec=TradingLogger)
        
        with pytest.raises(ValueError, match="Nieprawidłowy timeframe"):
            MT5Handler('EURUSD', timeframe='INVALID', logger=logger)

@pytest.mark.asyncio
async def test_invalid_volume(handler):
    """Test otwierania pozycji z nieprawidłowym wolumenem."""
    with patch('MetaTrader5.symbol_info') as mock_symbol_info, \
         patch('MetaTrader5.symbol_info_tick') as mock_tick:
        
        mock_symbol_info.return_value = Mock()
        mock_symbol_info.return_value.volume_min = 0.01
        mock_symbol_info.return_value.volume_max = 100.0
        mock_symbol_info.return_value.volume_step = 0.01
        
        mock_tick.return_value = Mock(
            bid=1.1000,
            ask=1.1002
        )

        # Test ujemnego wolumenu
        result = await handler.open_position(
            direction='BUY',
            volume=-1.0
        )
        assert result['status'] == 'error'
        assert 'Wolumen musi być większy od 0' in result['message']

        # Test zerowego wolumenu
        result = await handler.open_position(
            direction='BUY',
            volume=0.0
        )
        assert result['status'] == 'error'
        assert 'Wolumen musi być większy od 0' in result['message']

        # Test zbyt dużego wolumenu
        result = await handler.open_position(
            direction='BUY',
            volume=1001.0
        )
        assert result['status'] == 'error'
        assert 'Wolumen powyżej maksimum' in result['message']

@pytest.mark.asyncio
async def test_invalid_sl_tp(handler):
    """Test otwierania pozycji z nieprawidłowymi poziomami SL/TP."""
    current_price = 1.1000

    with patch('MetaTrader5.symbol_info_tick') as mock_tick, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info:
        
        mock_tick.return_value = Mock(
            bid=current_price,
            ask=current_price + 0.0002
        )
        mock_info = Mock()
        mock_info.volume_min = 0.01
        mock_info.volume_max = 100.0
        mock_info.volume_step = 0.01
        mock_symbol_info.return_value = mock_info

        # Test dla pozycji BUY
        result = await handler.open_position(
            direction='BUY',
            volume=0.1,
            stop_loss=current_price + 0.0100,  # SL powyżej ceny dla BUY
            take_profit=current_price - 0.0100  # TP poniżej ceny dla BUY
        )
        assert result['status'] == 'error'
        assert 'Dla pozycji BUY, stop loss musi być poniżej ceny wejścia' in result['message']

        # Test dla pozycji SELL
        result = await handler.open_position(
            direction='SELL',
            volume=0.1,
            stop_loss=current_price - 0.0100,  # SL poniżej ceny dla SELL
            take_profit=current_price + 0.0100  # TP powyżej ceny dla SELL
        )
        assert result['status'] == 'error'
        assert 'Dla pozycji SELL, stop loss musi być powyżej ceny wejścia' in result['message']

@pytest.mark.asyncio
async def test_invalid_direction(handler):
    """Test otwierania pozycji z nieprawidłowym kierunkiem."""
    result = await handler.open_position(
        direction='INVALID',
        volume=0.1
    )
    assert result['status'] == 'error'
    assert 'Nieprawidłowy kierunek' in result['message']

@pytest.mark.asyncio
async def test_symbol_min_volume(handler):
    """Test sprawdzania minimalnego wolumenu dla symbolu."""
    with patch('MetaTrader5.symbol_info') as mock_symbol_info, \
         patch('MetaTrader5.symbol_info_tick') as mock_tick:
        
        mock_info = Mock()
        mock_info.volume_min = 0.01
        mock_info.volume_max = 100.0
        mock_info.volume_step = 0.01
        mock_symbol_info.return_value = mock_info
        
        mock_tick.return_value = Mock(
            bid=1.1000,
            ask=1.1002
        )

        # Test wolumenu poniżej minimum
        result = await handler.open_position(
            direction='BUY',
            volume=0.001
        )
        assert result['status'] == 'error'
        assert 'Wolumen poniżej minimum' in result['message']

        # Test nieprawidłowego kroku wolumenu
        result = await handler.open_position(
            direction='BUY',
            volume=0.015
        )
        assert result['status'] == 'error'
        assert 'Nieprawidłowy krok wolumenu' in result['message']

@pytest.mark.asyncio
async def test_symbol_max_volume(handler):
    """Test sprawdzania maksymalnego wolumenu dla symbolu."""
    with patch('MetaTrader5.symbol_info') as mock_symbol_info, \
         patch('MetaTrader5.symbol_info_tick') as mock_tick:
        
        mock_symbol_info.return_value = Mock()
        mock_symbol_info.return_value.volume_max = 10.0
        mock_symbol_info.return_value.volume_min = 0.01
        mock_symbol_info.return_value.volume_step = 0.01
        
        mock_tick.return_value = Mock(
            bid=1.1000,
            ask=1.1002
        )

        result = await handler.open_position(
            direction='BUY',
            volume=11.0
        )
        assert result['status'] == 'error'
        assert 'Wolumen powyżej maksimum' in result['message']

@pytest.mark.asyncio
async def test_concurrent_operations(handler):
    """Test równoległego wykonywania operacji."""
    with patch('MetaTrader5.order_send') as mock_send, \
         patch('MetaTrader5.symbol_info_tick') as mock_tick, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info, \
         patch('MetaTrader5.copy_rates_from') as mock_copy:
        
        # Przygotuj mocki
        mock_tick.return_value = Mock(bid=1.1000, ask=1.1002)
        mock_info = Mock(volume_min=0.01, volume_max=100.0, volume_step=0.01)
        mock_symbol_info.return_value = mock_info
        mock_send.return_value = Mock(retcode=mt5.TRADE_RETCODE_DONE)
        
        # Przygotuj dane historyczne
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        historical_data = []
        for date in dates:
            historical_data.append({
                'time': int(date.timestamp()),
                'open': np.random.uniform(1.1000, 1.1100),
                'high': np.random.uniform(1.1100, 1.1200),
                'low': np.random.uniform(1.0900, 1.1000),
                'close': np.random.uniform(1.1000, 1.1100),
                'tick_volume': np.random.randint(1000, 2000),
                'spread': 2,
                'real_volume': np.random.randint(10000, 20000)
            })
        mock_copy.return_value = historical_data

        # Wykonaj równoległe operacje
        tasks = [
            handler.open_position(
                direction='BUY',
                volume=0.1,
                stop_loss=1.0950,
                take_profit=1.1050
            ),
            handler.get_current_price(),
            handler.get_historical_data(
                start_date=datetime(2024, 1, 1),
                num_bars=100
            )
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Sprawdź wyniki
        assert all(r is not None for r in results)
        assert isinstance(results[0], dict)  # Wynik otwarcia pozycji
        assert isinstance(results[1], dict)  # Aktualne ceny
        assert isinstance(results[2], pd.DataFrame)  # Dane historyczne

@pytest.mark.benchmark
def test_strategy_execution_speed(benchmark, handler):
    """Test wydajności wykonania operacji MT5."""
    with patch('MetaTrader5.symbol_info_tick') as mock_tick:
        mock_tick.return_value = Mock(
            bid=1.1000,
            ask=1.1002,
            last=1.1001,
            volume=1000,
            time=datetime.now().timestamp()
        )
        
        def get_price():
            """Funkcja do testowania wydajności."""
            return handler.get_current_price()
            
        result = benchmark(get_price)
        assert result is not None

@pytest.mark.memory
@pytest.mark.asyncio
async def test_memory_usage(handler):
    """Test zużycia pamięci."""
    with patch('MetaTrader5.copy_rates_from') as mock_copy:
        # Przygotuj duży zestaw danych
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
        historical_data = []
        for date in dates:
            historical_data.append({
                'time': int(date.timestamp()),
                'open': np.random.uniform(1.1000, 1.1100),
                'high': np.random.uniform(1.1100, 1.1200),
                'low': np.random.uniform(1.0900, 1.1000),
                'close': np.random.uniform(1.1000, 1.1100),
                'tick_volume': np.random.randint(1000, 2000),
                'spread': 2,
                'real_volume': np.random.randint(10000, 20000)
            })
        mock_copy.return_value = historical_data

        @memory_profiler.profile
        async def run_operations():
            """Funkcja do profilowania pamięci."""
            # Pobierz duży zestaw danych historycznych
            data = await handler.get_historical_data(
                start_date=datetime(2024, 1, 1),
                num_bars=1000
            )
            # Wykonaj operacje na danych
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            return data

        # Zmierz zużycie pamięci
        baseline = memory_profiler.memory_usage()[0]
        await run_operations()
        peak = max(memory_profiler.memory_usage())
        
        # Sprawdź czy zużycie pamięci jest poniżej 100MB
        assert peak - baseline < 100  # MB

@pytest.mark.integration
async def test_full_trading_cycle(handler):
    """Test pełnego cyklu tradingowego."""
    with patch('MetaTrader5.order_send') as mock_send, \
         patch('MetaTrader5.symbol_info_tick') as mock_tick, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info, \
         patch('MetaTrader5.copy_rates_from') as mock_copy, \
         patch('MetaTrader5.positions_get') as mock_positions:
        
        # Przygotuj mocki
        mock_tick.return_value = Mock(bid=1.1000, ask=1.1002)
        mock_info = Mock(volume_min=0.01, volume_max=100.0, volume_step=0.01)
        mock_symbol_info.return_value = mock_info
        
        # Symuluj udane otwarcie pozycji
        mock_send.return_value = Mock(
            retcode=mt5.TRADE_RETCODE_DONE,
            volume=0.1,
            price=1.1002,
            comment="Test trade"
        )
        
        # Symuluj dane historyczne
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        historical_data = []
        for date in dates:
            historical_data.append({
                'time': int(date.timestamp()),
                'open': np.random.uniform(1.1000, 1.1100),
                'high': np.random.uniform(1.1100, 1.1200),
                'low': np.random.uniform(1.0900, 1.1000),
                'close': np.random.uniform(1.1000, 1.1100),
                'tick_volume': np.random.randint(1000, 2000),
                'spread': 2,
                'real_volume': np.random.randint(10000, 20000)
            })
        mock_copy.return_value = historical_data
        
        # Symuluj otwartą pozycję
        mock_position = Mock(
            ticket=12345,
            type=mt5.ORDER_TYPE_BUY,
            volume=0.1,
            price_open=1.1002,
            price_current=1.1050,
            profit=48.0,
            sl=1.0950,
            tp=1.1100
        )
        mock_positions.return_value = [mock_position]
        
        # 1. Pobierz dane historyczne
        data = await handler.get_historical_data(
            start_date=datetime(2024, 1, 1),
            num_bars=100
        )
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        
        # 2. Pobierz aktualną cenę
        price = await handler.get_current_price()
        assert isinstance(price, dict)
        assert 'bid' in price and 'ask' in price
        
        # 3. Otwórz pozycję
        open_result = await handler.open_position(
            direction='BUY',
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100
        )
        assert open_result['status'] == 'success'
        assert open_result['volume'] == 0.1
        
        # 4. Pobierz otwarte pozycje
        positions = await handler.get_positions()
        assert len(positions) == 1
        assert positions[0]['ticket'] == 12345
        
        # 5. Zamknij pozycję
        mock_send.return_value = Mock(
            retcode=mt5.TRADE_RETCODE_DONE,
            volume=0.1,
            price=1.1050,
            comment="Test close"
        )
        
        close_result = await handler.close_position()
        assert close_result['status'] == 'success'
        assert close_result['volume'] == 0.1

@pytest.mark.asyncio
async def test_error_handling_under_load(handler):
    """Test obsługi błędów pod obciążeniem."""
    with patch('MetaTrader5.order_send') as mock_send, \
         patch('MetaTrader5.symbol_info_tick') as mock_tick, \
         patch('MetaTrader5.symbol_info') as mock_symbol_info:
        
        mock_tick.return_value = Mock(bid=1.1000, ask=1.1002)
        mock_info = Mock(volume_min=0.01, volume_max=100.0, volume_step=0.01)
        mock_symbol_info.return_value = mock_info
        
        # Symuluj różne błędy
        errors = [
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_CONNECTION,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_TIMEOUT,
            mt5.TRADE_RETCODE_ERROR
        ]
        
        mock_send.side_effect = [
            Mock(retcode=error, comment=f"Error {error}")
            for error in errors
        ]
        
        # Wykonaj wiele równoległych operacji
        tasks = []
        for _ in range(5):
            tasks.append(handler.open_position(
                direction='BUY',
                volume=0.1,
                stop_loss=1.0950,
                take_profit=1.1100
            ))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sprawdź czy wszystkie błędy zostały obsłużone
        for result in results:
            assert isinstance(result, dict)
            assert result['status'] == 'error'
            assert 'message' in result
            assert 'code' in result 