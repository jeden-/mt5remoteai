"""
Testy jednostkowe dla modułu performance_metrics.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtest.performance_metrics import PerformanceMetrics, TradeResult

@pytest.fixture
def sample_trades():
    """Przykładowe transakcje dla testów."""
    base_time = pd.Timestamp('2024-01-01')
    return [
        TradeResult(
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=1),
            entry_price=100.0,
            exit_price=105.0,
            direction='BUY',
            profit=500.0,
            size=1.0
        ),
        TradeResult(
            entry_time=base_time + timedelta(hours=2),
            exit_time=base_time + timedelta(hours=3),
            entry_price=105.0,
            exit_price=103.0,
            direction='SELL',
            profit=200.0,
            size=1.0
        ),
        TradeResult(
            entry_time=base_time + timedelta(hours=4),
            exit_time=base_time + timedelta(hours=5),
            entry_price=103.0,
            exit_price=101.0,
            direction='BUY',
            profit=-200.0,
            size=1.0
        )
    ]

@pytest.fixture
def sample_trades_all_profitable():
    """Przykładowe transakcje - wszystkie zyskowne."""
    base_time = pd.Timestamp('2024-01-01')
    return [
        TradeResult(
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=1),
            entry_price=100.0,
            exit_price=105.0,
            direction='BUY',
            profit=500.0,
            size=1.0
        ),
        TradeResult(
            entry_time=base_time + timedelta(hours=2),
            exit_time=base_time + timedelta(hours=3),
            entry_price=105.0,
            exit_price=110.0,
            direction='BUY',
            profit=500.0,
            size=1.0
        )
    ]

@pytest.fixture
def sample_trades_all_losing():
    """Przykładowe transakcje - wszystkie stratne."""
    base_time = pd.Timestamp('2024-01-01')
    return [
        TradeResult(
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=1),
            entry_price=100.0,
            exit_price=95.0,
            direction='BUY',
            profit=-500.0,
            size=1.0
        ),
        TradeResult(
            entry_time=base_time + timedelta(hours=2),
            exit_time=base_time + timedelta(hours=3),
            entry_price=95.0,
            exit_price=90.0,
            direction='BUY',
            profit=-500.0,
            size=1.0
        )
    ]

def test_trade_result():
    """Test klasy TradeResult."""
    trade = TradeResult(
        entry_time=pd.Timestamp('2024-01-01'),
        exit_time=pd.Timestamp('2024-01-01 01:00:00'),
        entry_price=100.0,
        exit_price=105.0,
        direction='BUY',
        profit=500.0,
        size=1.0
    )
    
    assert isinstance(trade.entry_time, pd.Timestamp)
    assert isinstance(trade.exit_time, pd.Timestamp)
    assert trade.entry_price == 100.0
    assert trade.exit_price == 105.0
    assert trade.direction == 'BUY'
    assert trade.profit == 500.0
    assert trade.size == 1.0

def test_initialization():
    """Test inicjalizacji PerformanceMetrics."""
    metrics = PerformanceMetrics([], 10000)
    assert metrics.trades == []
    assert metrics.initial_capital == 10000
    
    trades = [TradeResult(
        entry_time=pd.Timestamp('2024-01-01'),
        exit_time=pd.Timestamp('2024-01-01 01:00:00'),
        entry_price=100.0,
        exit_price=105.0,
        direction='BUY',
        profit=500.0,
        size=1.0
    )]
    metrics = PerformanceMetrics(trades, 20000)
    assert metrics.trades == trades
    assert metrics.initial_capital == 20000

def test_calculate_metrics_no_trades():
    """Test obliczania metryk dla pustej listy transakcji."""
    metrics = PerformanceMetrics([], 10000)
    results = metrics.calculate_metrics()
    
    assert results["total_return"] == 0
    assert results["win_rate"] == 0
    assert results["profit_factor"] == 0
    assert results["max_drawdown"] == 0
    assert results["sharpe_ratio"] == 0
    assert results["total_trades"] == 0
    assert results["avg_profit"] == 0
    assert results["avg_win"] == 0
    assert results["avg_loss"] == 0

def test_calculate_metrics_mixed_trades(sample_trades):
    """Test obliczania metryk dla mieszanych transakcji."""
    metrics = PerformanceMetrics(sample_trades, 10000)
    results = metrics.calculate_metrics()
    
    assert results["total_return"] == pytest.approx(5.0)  # (500 + 200 - 200) / 10000 * 100
    assert results["win_rate"] == pytest.approx(66.67, rel=1e-2)  # 2/3 * 100, tolerancja 1%
    assert results["profit_factor"] == pytest.approx(3.5)  # (500 + 200) / 200
    assert results["total_trades"] == 3
    assert results["avg_profit"] == pytest.approx(166.67, rel=1e-2)  # (500 + 200 - 200) / 3, tolerancja 1%
    assert results["avg_win"] == pytest.approx(350.0)  # (500 + 200) / 2
    assert results["avg_loss"] == -200.0

def test_calculate_metrics_all_profitable(sample_trades_all_profitable):
    """Test obliczania metryk dla samych zyskownych transakcji."""
    metrics = PerformanceMetrics(sample_trades_all_profitable, 10000)
    results = metrics.calculate_metrics()
    
    assert results["total_return"] == 10.0  # (500 + 500) / 10000 * 100
    assert results["win_rate"] == 100.0
    assert results["profit_factor"] == float('inf')  # brak strat
    assert results["total_trades"] == 2
    assert results["avg_profit"] == 500.0
    assert results["avg_win"] == 500.0
    assert results["avg_loss"] == 0.0

def test_calculate_metrics_all_losing(sample_trades_all_losing):
    """Test obliczania metryk dla samych stratnych transakcji."""
    metrics = PerformanceMetrics(sample_trades_all_losing, 10000)
    results = metrics.calculate_metrics()
    
    assert results["total_return"] == -10.0  # (-500 - 500) / 10000 * 100
    assert results["win_rate"] == 0.0
    assert results["profit_factor"] == 0.0  # brak zysków
    assert results["total_trades"] == 2
    assert results["avg_profit"] == -500.0
    assert results["avg_win"] == 0.0
    assert results["avg_loss"] == -500.0

def test_calculate_equity_curve(sample_trades):
    """Test obliczania krzywej kapitału."""
    metrics = PerformanceMetrics(sample_trades, 10000)
    equity_curve = metrics._calculate_equity_curve()
    
    assert len(equity_curve) == len(sample_trades) + 1  # początkowy kapitał + każda transakcja
    assert equity_curve[0] == 10000  # początkowy kapitał
    assert equity_curve[1] == 10500  # po pierwszej transakcji
    assert equity_curve[2] == 10700  # po drugiej transakcji
    assert equity_curve[3] == 10500  # po trzeciej transakcji

def test_calculate_max_drawdown():
    """Test obliczania maksymalnego drawdownu."""
    metrics = PerformanceMetrics([])
    
    # Test dla rosnącej krzywej
    equity_curve = np.array([10000, 10100, 10200, 10300])
    assert metrics._calculate_max_drawdown(equity_curve) == 0
    
    # Test dla spadającej krzywej
    equity_curve = np.array([10000, 9900, 9800, 9700])
    assert metrics._calculate_max_drawdown(equity_curve) == pytest.approx(0.03)  # (10000 - 9700) / 10000
    
    # Test dla krzywej ze spadkami i wzrostami
    equity_curve = np.array([10000, 9500, 9800, 9200, 9900])
    assert metrics._calculate_max_drawdown(equity_curve) == 0.08  # (10000 - 9200) / 10000

def test_calculate_sharpe_ratio():
    """Test obliczania wskaźnika Sharpe'a."""
    metrics = PerformanceMetrics([])
    
    # Test dla jednej wartości
    equity_curve = np.array([10000, 10100])
    assert metrics._calculate_sharpe_ratio(equity_curve) == 0
    
    # Test dla stałej krzywej
    equity_curve = np.array([10000, 10000, 10000])
    assert metrics._calculate_sharpe_ratio(equity_curve) == 0
    
    # Test dla rosnącej krzywej
    equity_curve = np.array([10000, 10100, 10200, 10300])
    sharpe = metrics._calculate_sharpe_ratio(equity_curve)
    assert sharpe > 0  # Powinien być dodatni dla rosnącej krzywej
    
    # Test dla spadającej krzywej
    equity_curve = np.array([10000, 9900, 9800, 9700])
    sharpe = metrics._calculate_sharpe_ratio(equity_curve)
    assert sharpe < 0  # Powinien być ujemny dla spadającej krzywej 

def test_calculate_max_drawdown_rising():
    """Test obliczania maksymalnego drawdownu dla rosnącej krzywej kapitału."""
    metrics = PerformanceMetrics([])
    equity_curve = np.array([1000, 1100, 1200, 1300, 1400])
    
    max_dd = metrics._calculate_max_drawdown(equity_curve)
    
    assert max_dd == 0.0

def test_calculate_max_drawdown_falling():
    """Test obliczania maksymalnego drawdownu dla spadającej krzywej kapitału."""
    metrics = PerformanceMetrics([])
    equity_curve = np.array([1000, 900, 800, 700, 600])
    
    max_dd = metrics._calculate_max_drawdown(equity_curve)
    
    assert max_dd == 0.4  # (1000 - 600) / 1000

def test_calculate_max_drawdown_multiple_peaks():
    """Test obliczania maksymalnego drawdownu dla krzywej z wieloma szczytami."""
    metrics = PerformanceMetrics([])
    # Krzywa: wzrost -> spadek -> wzrost -> większy spadek -> wzrost
    equity_curve = np.array([1000, 1100, 900, 1200, 800, 1000])
    
    max_dd = metrics._calculate_max_drawdown(equity_curve)
    
    # Największy drawdown to spadek z 1200 do 800
    assert max_dd == pytest.approx(0.3333, rel=1e-3)  # (1200 - 800) / 1200

def test_calculate_max_drawdown_single_value():
    """Test obliczania maksymalnego drawdownu dla pojedynczej wartości."""
    metrics = PerformanceMetrics([])
    equity_curve = np.array([1000])
    
    max_dd = metrics._calculate_max_drawdown(equity_curve)
    
    assert max_dd == 0.0

def test_calculate_sharpe_ratio_different_risk_free():
    """Test obliczania wskaźnika Sharpe'a dla różnych stóp wolnych od ryzyka."""
    metrics = PerformanceMetrics([])
    equity_curve = np.array([1000, 1100, 1200, 1300, 1400])
    
    # Dla niskiej stopy wolnej od ryzyka
    sharpe_low = metrics._calculate_sharpe_ratio(equity_curve, risk_free_rate=0.01)
    # Dla wysokiej stopy wolnej od ryzyka
    sharpe_high = metrics._calculate_sharpe_ratio(equity_curve, risk_free_rate=0.10)
    
    assert sharpe_low > sharpe_high  # Wyższy Sharpe dla niższej stopy wolnej od ryzyka

def test_calculate_sharpe_ratio_zero_std():
    """Test obliczania wskaźnika Sharpe'a gdy odchylenie standardowe jest zero."""
    metrics = PerformanceMetrics([])
    # Stała wartość kapitału = brak zmienności
    equity_curve = np.array([1000, 1000, 1000, 1000])
    
    sharpe = metrics._calculate_sharpe_ratio(equity_curve)
    
    assert sharpe == 0.0

def test_calculate_sharpe_ratio_short_data():
    """Test obliczania wskaźnika Sharpe'a dla krótkiej serii danych."""
    metrics = PerformanceMetrics([])
    # Trzy wartości - minimum dla obliczenia odchylenia standardowego
    equity_curve = np.array([1000, 1100, 1200])
    
    sharpe = metrics._calculate_sharpe_ratio(equity_curve)
    
    assert isinstance(sharpe, float)
    assert sharpe > 0  # Powinien być dodatni dla rosnącej krzywej

def test_calculate_sharpe_ratio_single_value():
    """Test obliczania wskaźnika Sharpe'a dla pojedynczej wartości."""
    metrics = PerformanceMetrics([])
    equity_curve = np.array([1000])
    
    sharpe = metrics._calculate_sharpe_ratio(equity_curve)
    
    assert sharpe == 0.0

def test_calculate_metrics_zero_gross_loss():
    """Test obliczania metryk gdy nie ma strat."""
    base_time = pd.Timestamp('2024-01-01')
    trades = [
        TradeResult(
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=1),
            entry_price=100.0,
            exit_price=110.0,
            direction='BUY',
            profit=1000.0,
            size=1.0
        ),
        TradeResult(
            entry_time=base_time + timedelta(hours=2),
            exit_time=base_time + timedelta(hours=3),
            entry_price=110.0,
            exit_price=120.0,
            direction='BUY',
            profit=1000.0,
            size=1.0
        )
    ]
    
    metrics = PerformanceMetrics(trades)
    results = metrics.calculate_metrics()
    
    assert results['win_rate'] == 100.0
    assert results['profit_factor'] == float('inf')
    assert results['avg_loss'] == 0.0

def test_calculate_metrics_zero_profitable_trades():
    """Test obliczania metryk gdy nie ma zyskownych transakcji."""
    base_time = pd.Timestamp('2024-01-01')
    trades = [
        TradeResult(
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=1),
            entry_price=100.0,
            exit_price=90.0,
            direction='BUY',
            profit=-1000.0,
            size=1.0
        ),
        TradeResult(
            entry_time=base_time + timedelta(hours=2),
            exit_time=base_time + timedelta(hours=3),
            entry_price=90.0,
            exit_price=80.0,
            direction='BUY',
            profit=-1000.0,
            size=1.0
        )
    ]
    
    metrics = PerformanceMetrics(trades)
    results = metrics.calculate_metrics()
    
    assert results['win_rate'] == 0.0
    assert results['profit_factor'] == 0.0
    assert results['avg_win'] == 0.0

def test_calculate_metrics_many_trades():
    """Test obliczania metryk dla dużej liczby transakcji."""
    base_time = pd.Timestamp('2024-01-01')
    trades = []
    
    # Generujemy 1000 transakcji
    for i in range(1000):
        profit = 100.0 if i % 2 == 0 else -50.0  # Na przemian zyski i straty
        trades.append(
            TradeResult(
                entry_time=base_time + timedelta(hours=i),
                exit_time=base_time + timedelta(hours=i+1),
                entry_price=100.0,
                exit_price=100.0 + profit,
                direction='BUY' if profit > 0 else 'SELL',
                profit=profit,
                size=1.0
            )
        )
    
    metrics = PerformanceMetrics(trades)
    results = metrics.calculate_metrics()
    
    assert results['total_trades'] == 1000
    assert results['win_rate'] == 50.0
    assert results['avg_win'] == 100.0
    assert results['avg_loss'] == -50.0
    assert results['profit_factor'] == 2.0  # (500 * 100) / (500 * 50) 