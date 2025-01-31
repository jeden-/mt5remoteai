"""
Moduł zawierający testy dla DemoTestRunner.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from src.demo_test.demo_runner import DemoTestRunner
from src.utils.logger import TradingLogger
from src.strategies.basic_strategy import BasicStrategy


@pytest.fixture
def mock_strategy():
    """Fixture tworzący zamockowaną strategię."""
    strategy_mock = Mock(spec=BasicStrategy)
    strategy_mock.mt5 = Mock()
    
    # Symulacja zmiany salda w czasie
    account_info_values = [
        {'balance': 10000.0},  # Saldo początkowe
        {'balance': 10050.0},  # Po zyskownej transakcji
        {'balance': 9970.0}    # Po stratnej transakcji
    ]
    strategy_mock.mt5.get_account_info.side_effect = account_info_values
    
    strategy_mock.analyze_market = AsyncMock(return_value={
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1
        },
        'technical_indicators': {
            'sma_20': 1.09,
            'sma_50': 1.08
        },
        'ollama_analysis': 'RECOMMENDATION: BUY\nStrength: High',
        'claude_analysis': 'Rekomendacja: long\nSugerowany SL: 20 pips\nSugerowany TP: 60 pips'
    })
    strategy_mock.generate_signals = AsyncMock(return_value={
        'symbol': 'EURUSD',
        'action': 'BUY',
        'entry_price': 1.1,
        'stop_loss': 1.09,
        'take_profit': 1.12,
        'volume': 0.1
    })
    strategy_mock.execute_signals = AsyncMock(return_value={
        'ticket': 123,
        'symbol': 'EURUSD',
        'type': 'BUY',
        'volume': 0.1,
        'price': 1.1,
        'sl': 1.09,
        'tp': 1.12,
        'profit': 50.0
    })
    return strategy_mock


@pytest.fixture
def mock_strategy_losing():
    """Fixture tworzący zamockowaną strategię ze stratnymi transakcjami."""
    strategy_mock = Mock(spec=BasicStrategy)
    strategy_mock.mt5 = Mock()
    
    # Symulacja spadku salda
    account_info_values = [
        {'balance': 10000.0},  # Saldo początkowe
        {'balance': 9970.0}    # Po stratnej transakcji
    ]
    strategy_mock.mt5.get_account_info.side_effect = account_info_values
    
    strategy_mock.analyze_market = AsyncMock(return_value={
        'market_data': {
            'symbol': 'EURUSD',
            'current_price': 1.1
        }
    })
    strategy_mock.generate_signals = AsyncMock(return_value={
        'symbol': 'EURUSD',
        'action': 'BUY',
        'entry_price': 1.1,
        'stop_loss': 1.09,
        'take_profit': 1.12,
        'volume': 0.1
    })
    strategy_mock.execute_signals = AsyncMock(return_value={
        'ticket': 123,
        'symbol': 'EURUSD',
        'type': 'BUY',
        'volume': 0.1,
        'price': 1.1,
        'sl': 1.09,
        'tp': 1.12,
        'profit': -30.0
    })
    return strategy_mock


@pytest.fixture
def mock_logger():
    """Fixture tworzący zamockowany logger."""
    return Mock(spec=TradingLogger)


@pytest.fixture
def config():
    """Fixture tworzący konfigurację."""
    return {
        'max_position_size': 0.01,
        'max_risk_per_trade': 0.005,
        'allowed_symbols': ['EURUSD']
    }


@pytest.fixture
def demo_runner(mock_strategy, mock_logger, config):
    """Fixture tworzący DemoTestRunner."""
    return DemoTestRunner(mock_strategy, mock_logger, config)


@pytest.mark.asyncio
async def test_initialize_test(demo_runner, mock_strategy, mock_logger):
    """Test inicjalizacji testu demo."""
    await demo_runner.initialize_test()
    
    assert demo_runner.start_balance == 10000.0
    assert demo_runner.current_balance == 10000.0
    mock_logger.log_trade.assert_called_once()


@pytest.mark.asyncio
async def test_run_single_symbol_test(demo_runner, mock_strategy, mock_logger):
    """Test pojedynczego testu na symbolu."""
    await demo_runner.initialize_test()
    results = await demo_runner.run_single_symbol_test('EURUSD', duration_minutes=1)
    
    assert results['symbol'] == 'EURUSD'
    assert results['trades_count'] > 0
    assert results['profitable_trades'] > 0
    assert results['win_rate'] > 0
    assert results['profit'] > 0
    
    mock_strategy.analyze_market.assert_called()
    mock_strategy.generate_signals.assert_called()
    mock_strategy.execute_signals.assert_called()
    mock_logger.log_trade.assert_called()
    mock_logger.log_ai_analysis.assert_called()


@pytest.mark.asyncio
async def test_run_single_symbol_test_error(demo_runner, mock_strategy, mock_logger):
    """Test obsługi błędów w pojedynczym teście."""
    await demo_runner.initialize_test()
    mock_strategy.analyze_market.side_effect = Exception("Test error")
    
    results = await demo_runner.run_single_symbol_test('EURUSD', duration_minutes=1)
    
    assert results['trades_count'] == 0
    assert results['profitable_trades'] == 0
    assert results['win_rate'] == 0
    mock_logger.log_error.assert_called()


@pytest.mark.asyncio
async def test_run_full_test(demo_runner, mock_strategy, mock_logger):
    """Test pełnego zestawu testów."""
    results = await demo_runner.run_full_test(['EURUSD', 'GBPUSD'], duration_minutes=1)
    
    assert len(results) == 2
    assert all(r['symbol'] in ['EURUSD', 'GBPUSD'] for r in results)
    assert all(r['trades_count'] >= 0 for r in results)
    
    mock_logger.log_trade.assert_called()


def test_generate_test_report(demo_runner):
    """Test generowania raportu."""
    demo_runner.start_balance = 10000.0
    demo_runner.current_balance = 10500.0
    demo_runner.test_results = [{
        'symbol': 'EURUSD',
        'trades_count': 10,
        'profitable_trades': 7,
        'win_rate': 70.0,
        'profit': 500.0
    }]
    
    report = demo_runner.generate_test_report()
    
    assert isinstance(report, str)
    assert "RAPORT Z TESTÓW DEMO" in report
    assert "EURUSD" in report
    assert "Saldo początkowe: 10000.0" in report
    assert "Saldo końcowe: 10500.0" in report
    assert "Procent zwrotu: 5.0%" in report


@pytest.mark.asyncio
async def test_run_single_symbol_test_no_trades(demo_runner, mock_strategy, mock_logger):
    """Test zachowania gdy nie ma żadnych transakcji."""
    # Mockujemy analyze_market aby nie generował wyjątków
    mock_strategy.analyze_market = AsyncMock(return_value={
        'market_data': {'symbol': 'EURUSD'},
        'technical_indicators': {},
        'ollama_analysis': '',
        'claude_analysis': ''
    })
    
    # Mockujemy brak sygnałów tradingowych
    mock_strategy.generate_signals = AsyncMock(return_value={
        'symbol': 'EURUSD',
        'action': 'WAIT',
        'entry_price': None,
        'stop_loss': None,
        'take_profit': None,
        'volume': 0
    })
    
    # Mockujemy brak wykonanych transakcji
    mock_strategy.execute_signals = AsyncMock(return_value=None)
    
    # Mockujemy niezmienione saldo - ważne aby zwracać tę samą wartość za każdym razem
    initial_balance = {'balance': 10000.0}
    mock_strategy.mt5.get_account_info = Mock(return_value=initial_balance)
    
    # Inicjalizacja testu
    await demo_runner.initialize_test()
    
    # Ustawiamy krótki czas testu (1 minuta)
    results = await demo_runner.run_single_symbol_test('EURUSD', duration_minutes=1)
    
    # Sprawdzamy czy nie było żadnych transakcji
    assert results['trades_count'] == 0
    assert results['profitable_trades'] == 0
    assert results['win_rate'] == 0
    assert results['profit'] == 0
    assert results['profit_percentage'] == 0
    assert results['start_balance'] == initial_balance['balance']
    assert results['end_balance'] == initial_balance['balance']
    
    # Sprawdzamy czy saldo nie zmieniło się w trakcie testu
    assert demo_runner.start_balance == initial_balance['balance']
    assert demo_runner.current_balance == initial_balance['balance']
    
    # Sprawdzamy czy metody były wywołane
    mock_strategy.analyze_market.assert_called()
    mock_strategy.generate_signals.assert_called()
    mock_strategy.execute_signals.assert_not_called()  # Nie powinno być wywołane dla WAIT
    mock_strategy.mt5.get_account_info.assert_called()


@pytest.mark.asyncio
async def test_run_single_symbol_test_losing_trades(demo_runner, mock_strategy_losing, mock_logger):
    """Test zachowania przy stratnych transakcjach."""
    demo_runner.strategy = mock_strategy_losing
    await demo_runner.initialize_test()
    
    results = await demo_runner.run_single_symbol_test('EURUSD', duration_minutes=1)
    
    assert results['trades_count'] > 0
    assert results['profitable_trades'] == 0
    assert results['win_rate'] == 0
    assert results['profit'] < 0
    assert results['profit_percentage'] < 0
    assert results['end_balance'] < results['start_balance']


@pytest.mark.asyncio
async def test_run_full_test_invalid_symbol(demo_runner, mock_strategy, mock_logger):
    """Test zachowania przy nieprawidłowym symbolu."""
    mock_strategy.analyze_market.side_effect = Exception("Invalid symbol")
    
    results = await demo_runner.run_full_test(['INVALID'], duration_minutes=1)
    
    assert len(results) == 1
    assert results[0]['trades_count'] == 0
    assert results[0]['profitable_trades'] == 0
    mock_logger.log_error.assert_called()


def test_generate_test_report_no_trades(demo_runner):
    """Test generowania raportu gdy nie było transakcji."""
    demo_runner.start_balance = 10000.0
    demo_runner.current_balance = 10000.0
    demo_runner.test_results = [{
        'symbol': 'EURUSD',
        'trades_count': 0,
        'profitable_trades': 0,
        'win_rate': 0.0,
        'profit': 0
    }]
    
    report = demo_runner.generate_test_report()
    
    assert "Liczba transakcji: 0" in report
    assert "Win rate: 0.0%" in report
    assert "Zysk: 0" in report


def test_generate_test_report_multiple_symbols(demo_runner):
    """Test generowania raportu dla wielu symboli."""
    demo_runner.start_balance = 10000.0
    demo_runner.current_balance = 10800.0
    demo_runner.test_results = [
        {
            'symbol': 'EURUSD',
            'trades_count': 10,
            'profitable_trades': 7,
            'win_rate': 70.0,
            'profit': 500.0
        },
        {
            'symbol': 'GBPUSD',
            'trades_count': 8,
            'profitable_trades': 5,
            'win_rate': 62.5,
            'profit': 300.0
        }
    ]
    
    report = demo_runner.generate_test_report()
    
    assert "EURUSD" in report
    assert "GBPUSD" in report
    assert "Całkowity zysk/strata: 800.0" in report
    assert "Procent zwrotu: 8.0%" in report


@pytest.mark.asyncio
async def test_initialize_test_no_balance(demo_runner, mock_strategy, mock_logger):
    """Test inicjalizacji gdy nie można pobrać salda."""
    # Mockujemy brak salda
    mock_strategy.mt5.get_account_info.return_value = {}
    
    await demo_runner.initialize_test()
    
    # Sprawdzamy czy ustawiono domyślne saldo
    assert demo_runner.start_balance == 10000.0
    assert demo_runner.current_balance == 10000.0
    mock_logger.log_trade.assert_called_once()


@pytest.mark.asyncio
async def test_run_full_test_with_error(demo_runner, mock_strategy, mock_logger):
    """Test pełnego testu z błędem dla jednego symbolu."""
    # Mockujemy błąd dla drugiego symbolu
    async def mock_run_single_symbol_test(symbol, duration_minutes):
        if symbol == 'GBPUSD':
            raise Exception("Test error")
        return {
            'symbol': symbol,
            'duration_minutes': duration_minutes,
            'trades_count': 1,
            'profitable_trades': 1,
            'win_rate': 100.0,
            'start_balance': 10000.0,
            'end_balance': 10050.0,
            'profit': 50.0,
            'profit_percentage': 0.5
        }
    
    demo_runner.run_single_symbol_test = mock_run_single_symbol_test
    
    results = await demo_runner.run_full_test(['EURUSD', 'GBPUSD'], duration_minutes=1)
    
    assert len(results) == 2
    assert results[0]['symbol'] == 'EURUSD'
    assert results[0]['trades_count'] == 1
    assert results[1]['symbol'] == 'GBPUSD'
    assert results[1]['trades_count'] == 0
    assert results[1]['profit'] == 0
    
    # Sprawdzamy czy zalogowano błąd
    mock_logger.log_error.assert_called_once()
    mock_logger.log_trade.assert_called()


@pytest.mark.asyncio
async def test_run_full_test_multiple_symbols(demo_runner, mock_strategy, mock_logger):
    """Test pełnego testu na wielu symbolach."""
    results = await demo_runner.run_full_test(['EURUSD', 'GBPUSD', 'USDJPY'], duration_minutes=1)
    
    assert len(results) == 3
    assert all(r['symbol'] in ['EURUSD', 'GBPUSD', 'USDJPY'] for r in results)
    
    # Sprawdzamy czy zalogowano start i koniec dla każdego symbolu
    assert mock_logger.log_trade.call_count >= 6  # min. 2 logi na symbol 