"""
Testy jednostkowe dla modułu visualizer.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from src.backtest.visualizer import BacktestVisualizer
from src.backtest.performance_metrics import TradeResult

@pytest.fixture
def sample_data():
    """Fixture dostarczający przykładowe dane historyczne."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.randint(1000, 5000, 100),
        'SMA_20': np.random.normal(100, 0.5, 100),
        'SMA_50': np.random.normal(100, 0.3, 100)
    }, index=dates)
    
    # Upewniamy się, że high jest najwyższy a low najniższy
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
    
    return data

@pytest.fixture
def sample_trades():
    """Fixture dostarczający przykładowe transakcje."""
    base_time = pd.Timestamp('2024-01-01 10:00:00')
    trades = [
        TradeResult(
            entry_time=base_time,
            exit_time=base_time + timedelta(hours=1),
            entry_price=100.0,
            exit_price=110.0,
            direction='BUY',
            profit=100.0,
            size=1.0
        ),
        TradeResult(
            entry_time=base_time + timedelta(hours=2),
            exit_time=base_time + timedelta(hours=3),
            entry_price=110.0,
            exit_price=105.0,
            direction='BUY',
            profit=-50.0,
            size=1.0
        ),
        TradeResult(
            entry_time=base_time + timedelta(hours=4),
            exit_time=base_time + timedelta(hours=5),
            entry_price=105.0,
            exit_price=115.0,
            direction='SELL',
            profit=100.0,
            size=1.0
        )
    ]
    return trades

def test_initialization(sample_data, sample_trades):
    """Test inicjalizacji klasy BacktestVisualizer."""
    visualizer = BacktestVisualizer(sample_data, sample_trades)
    assert visualizer.data.equals(sample_data)
    assert visualizer.trades == sample_trades

def test_calculate_equity_curve(sample_data, sample_trades):
    """Test obliczania krzywej kapitału."""
    visualizer = BacktestVisualizer(sample_data, sample_trades)
    equity_curve = visualizer._calculate_equity_curve()
    
    assert isinstance(equity_curve, pd.Series)
    assert len(equity_curve) == len(sample_data)
    assert equity_curve.iloc[0] == 10000.0  # początkowy kapitał
    assert equity_curve.iloc[-1] == 10150.0  # końcowy kapitał (10000 + 100 - 50 + 100)

def test_create_dashboard(sample_data, sample_trades):
    """Test tworzenia dashboardu."""
    visualizer = BacktestVisualizer(sample_data, sample_trades)
    fig = visualizer.create_dashboard()
    
    assert isinstance(fig, go.Figure)
    # 1 świeczki + 2 SMA + 6 punkty wejścia/wyjścia + 1 krzywa kapitału + 1 histogram
    assert len(fig.data) == 11
    assert fig.layout.height == 1200
    assert fig.layout.title.text == "Dashboard Wyników Backtestu"
    
    # Sprawdź czy wszystkie wykresy są na swoich miejscach
    candlestick_data = [trace for trace in fig.data if isinstance(trace, go.Candlestick)][0]
    assert candlestick_data.name == 'Cena'
    
    sma_traces = [trace for trace in fig.data if isinstance(trace, go.Scatter) and 'SMA' in trace.name]
    assert len(sma_traces) == 2
    assert 'SMA 20' in [trace.name for trace in sma_traces]
    assert 'SMA 50' in [trace.name for trace in sma_traces]

def test_save_dashboard(sample_data, sample_trades, tmp_path):
    """Test zapisywania dashboardu do pliku."""
    visualizer = BacktestVisualizer(sample_data, sample_trades)
    output_file = tmp_path / "test_dashboard.html"
    
    visualizer.save_dashboard(str(output_file))
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_empty_trades(sample_data):
    """Test wizualizacji dla pustej listy transakcji."""
    visualizer = BacktestVisualizer(sample_data, [])
    fig = visualizer.create_dashboard()
    
    assert isinstance(fig, go.Figure)
    # 1 świeczki + 2 SMA + 1 krzywa kapitału + 1 histogram
    assert len(fig.data) == 5
    
    equity_curve = visualizer._calculate_equity_curve()
    assert all(equity_curve == 10000.0)  # kapitał powinien być stały

def test_single_trade(sample_data):
    """Test wizualizacji dla pojedynczej transakcji."""
    trade = TradeResult(
        entry_time=pd.Timestamp('2024-01-01 10:00:00'),
        exit_time=pd.Timestamp('2024-01-01 11:00:00'),
        entry_price=100.0,
        exit_price=110.0,
        direction='BUY',
        profit=100.0,
        size=1.0
    )
    
    visualizer = BacktestVisualizer(sample_data, [trade])
    fig = visualizer.create_dashboard()
    
    assert isinstance(fig, go.Figure)
    # 1 świeczki + 2 SMA + 2 punkty (wejście/wyjście) + 1 krzywa kapitału + 1 histogram
    assert len(fig.data) == 7

def test_invalid_data():
    """Test obsługi nieprawidłowych danych."""
    with pytest.raises(KeyError, match='open'):
        # Brak wymaganych kolumn
        invalid_data = pd.DataFrame({'price': [100, 101, 102]})
        visualizer = BacktestVisualizer(invalid_data, [])
        visualizer.create_dashboard()  # Powinno rzucić wyjątek przy próbie dostępu do nieistniejących kolumn 

def test_invalid_trades():
    """Test obsługi nieprawidłowych transakcji."""
    data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [98, 99, 100],
        'close': [101, 102, 103],
        'volume': [1000, 1100, 1200],
        'SMA_20': [100, 101, 102],
        'SMA_50': [99, 100, 101]
    }, index=pd.date_range(start='2024-01-01', periods=3, freq='h'))
    
    # Transakcja z datą spoza zakresu danych
    invalid_trade = TradeResult(
        entry_time=pd.Timestamp('2025-01-01'),
        exit_time=pd.Timestamp('2025-01-02'),
        entry_price=100.0,
        exit_price=110.0,
        direction='BUY',
        profit=100.0,
        size=1.0
    )
    
    visualizer = BacktestVisualizer(data, [invalid_trade])
    fig = visualizer.create_dashboard()
    
    assert isinstance(fig, go.Figure)
    # Punkty wejścia/wyjścia nie powinny być dodane dla transakcji poza zakresem
    assert len([trace for trace in fig.data if trace.name in ['Wejścia', 'Wyjścia']]) == 0

def test_missing_technical_indicators(sample_trades):
    """Test wizualizacji bez wskaźników technicznych."""
    data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [98, 99, 100],
        'close': [101, 102, 103],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range(start='2024-01-01', periods=3, freq='h'))
    
    visualizer = BacktestVisualizer(data, sample_trades)
    fig = visualizer.create_dashboard(show_indicators=False)
    
    assert isinstance(fig, go.Figure)
    # Nie powinno być wykresów SMA
    assert len([trace for trace in fig.data if 'SMA' in str(trace.name)]) == 0

def test_dashboard_layout_customization(sample_data, sample_trades):
    """Test dostosowania wyglądu dashboardu."""
    visualizer = BacktestVisualizer(sample_data, sample_trades)
    fig = visualizer.create_dashboard()
    
    # Sprawdź ustawienia układu
    assert fig.layout.xaxis.rangeslider.visible == False  # Poprawiona składnia
    assert fig.layout.showlegend == True
    assert len(fig.layout.annotations) == 3  # tytuły podwykresów
    assert fig.layout.annotations[0].text == 'Cena i Transakcje'
    assert fig.layout.annotations[1].text == 'Krzywa Kapitału'
    assert fig.layout.annotations[2].text == 'Rozkład Zysków/Strat' 