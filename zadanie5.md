ZADANIE #5 - Implementacja modułu backtestingu

1. Utwórz nowy folder src/backtest/ i dodaj następujące pliki:

src/backtest/
├── __init__.py
├── backtester.py
├── data_loader.py
├── performance_metrics.py
└── visualizer.py

2. W pliku src/backtest/data_loader.py zaimplementuj:
```python
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, List, Optional
import numpy as np

class HistoricalDataLoader:
    def __init__(self, symbol: str, timeframe: str = "1H", start_date: Optional[datetime] = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_map = {
            "1M": mt5.TIMEFRAME_M1,
            "5M": mt5.TIMEFRAME_M5,
            "15M": mt5.TIMEFRAME_M15,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1,
        }
        self.start_date = start_date or (datetime.now() - timedelta(days=30))
        
    def load_data(self) -> pd.DataFrame:
        """Ładuje historyczne dane z MT5"""
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MT5")
            
        try:
            # Pobierz dane historyczne
            rates = mt5.copy_rates_from(
                self.symbol,
                self.timeframe_map[self.timeframe],
                self.start_date,
                10000  # maksymalna liczba świec
            )
            
            if rates is None:
                raise RuntimeError(f"Failed to get historical data for {self.symbol}")
                
            # Konwertuj na DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Dodaj podstawowe wskaźniki
            df = self.add_indicators(df)
            
            return df
            
        finally:
            mt5.shutdown()
            
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje wskaźniki techniczne do danych"""
        # SMA
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (std * 2)
        df['BB_lower'] = df['BB_middle'] - (std * 2)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df

3. W pliku src/backtest/performance_metrics.py zaimplementuj:
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str  # 'BUY' or 'SELL'
    profit: float
    size: float

class PerformanceMetrics:
    def __init__(self, trades: List[TradeResult], initial_capital: float = 10000):
        self.trades = trades
        self.initial_capital = initial_capital
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Oblicza metryki wydajności strategii"""
        if not self.trades:
            return {
                "total_return": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "total_trades": 0
            }
            
        # Podstawowe metryki
        profitable_trades = len([t for t in self.trades if t.profit > 0])
        total_trades = len(self.trades)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Zyski i straty
        gross_profit = sum(t.profit for t in self.trades if t.profit > 0)
        gross_loss = abs(sum(t.profit for t in self.trades if t.profit < 0))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Krzywa kapitału
        equity_curve = self._calculate_equity_curve()
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        
        return {
            "total_return": (equity_curve[-1] - self.initial_capital) / self.initial_capital * 100,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown * 100,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades
        }
        
    def _calculate_equity_curve(self) -> np.ndarray:
        """Oblicza krzywą kapitału"""
        equity = [self.initial_capital]
        for trade in self.trades:
            equity.append(equity[-1] + trade.profit)
        return np.array(equity)
        
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Oblicza maksymalny drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _calculate_sharpe_ratio(self, equity_curve: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Oblicza wskaźnik Sharpe'a"""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        excess_returns = returns - risk_free_rate/252  # dzienny risk-free rate
        
        if len(excess_returns) < 2:
            return 0
            
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0

4. W pliku src/backtest/backtester.py zaimplementuj:
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from .data_loader import HistoricalDataLoader
from .performance_metrics import PerformanceMetrics, TradeResult
from ..strategies.basic_strategy import BasicStrategy
from ..utils.logger import TradingLogger

class Backtester:
    def __init__(
        self,
        strategy: BasicStrategy,
        symbol: str,
        timeframe: str = "1H",
        initial_capital: float = 10000,
        start_date: Optional[datetime] = None,
        logger: Optional[TradingLogger] = None
    ):
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.logger = logger
        self.trades: List[TradeResult] = []
        
    async def run_backtest(self) -> Dict[str, float]:
        """Przeprowadza backtesting strategii"""
        # Załaduj dane historyczne
        data_loader = HistoricalDataLoader(self.symbol, self.timeframe, self.start_date)
        df = data_loader.load_data()
        
        current_position = None
        
        for i in range(len(df)-1):
            current_data = df.iloc[:i+1]
            next_bar = df.iloc[i+1]
            
            # Przygotuj dane dla strategii
            market_data = {
                'symbol': self.symbol,
                'current_price': current_data['close'].iloc[-1],
                'sma_20': current_data['SMA_20'].iloc[-1],
                'sma_50': current_data['SMA_50'].iloc[-1],
                'rsi': current_data['RSI'].iloc[-1],
                'macd': current_data['MACD'].iloc[-1],
                'signal_line': current_data['Signal_Line'].iloc[-1]
            }
            
            # Generuj sygnały
            signals = await self.strategy.generate_signals({'market_data': market_data})
            
            # Wykonaj transakcje
            if signals['action'] != 'WAIT' and current_position is None:
                # Otwórz pozycję
                current_position = {
                    'entry_time': current_data.index[-1],
                    'entry_price': next_bar['open'],
                    'direction': signals['action'],
                    'size': signals['position_size']
                }
                
            elif current_position is not None:
                # Sprawdź warunki zamknięcia
                if self._should_close_position(current_position, next_bar, signals):
                    # Zamknij pozycję
                    trade = TradeResult(
                        entry_time=current_position['entry_time'],
                        exit_time=next_bar.name,
                        entry_price=current_position['entry_price'],
                        exit_price=next_bar['open'],
                        direction=current_position['direction'],
                        profit=self._calculate_profit(current_position, next_bar['open']),
                        size=current_position['size']
                    )
                    self.trades.append(trade)
                    current_position = None
                    
                    if self.logger:
                        self.logger.log_trade({
                            'type': 'backtest',
                            'trade': trade.__dict__
                        })
                        
        # Oblicz metryki wydajności
        metrics = PerformanceMetrics(self.trades, self.initial_capital)
        return metrics.calculate_metrics()
        
    def _should_close_position(self, position: Dict, current_bar: pd.Series, signals: Dict) -> bool:
        """Sprawdza, czy należy zamknąć pozycję"""
        if position['direction'] == 'BUY':
            return (
                current_bar['low'] <= signals.get('stop_loss', 0) or
                current_bar['high'] >= signals.get('take_profit', float('inf'))
            )
        else:  # SELL
            return (
                current_bar['high'] >= signals.get('stop_loss', float('inf')) or
                current_bar['low'] <= signals.get('take_profit', 0)
            )
            
    def _calculate_profit(self, position: Dict, exit_price: float) -> float:
        """Oblicza zysk/stratę z transakcji"""
        if position['direction'] == 'BUY':
            return (exit_price - position['entry_price']) * position['size']
        else:  # SELL
            return (position['entry_price'] - exit_price) * position['size']

5. W pliku src/backtest/visualizer.py zaimplementuj:
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict
from .performance_metrics import TradeResult

class BacktestVisualizer:
    def __init__(self, data: pd.DataFrame, trades: List[TradeResult]):
        self.data = data
        self.trades = trades
        
    def create_dashboard(self) -> go.Figure:
        """Tworzy interaktywny dashboard z wynikami backtestu"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Trades', 'Equity Curve', 'Trade Distribution')
        )
        
        # Wykres ceny i transakcji
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['open'],
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Dodaj punkty wejścia/wyjścia
        for trade in self.trades:
            color = 'green' if trade.profit > 0 else 'red'
            # Punkt wejścia
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time],
                    y=[trade.entry_price],
                    mode='markers',
                    marker=dict(size=10, symbol='triangle-up' if trade.direction == 'BUY' else 'triangle-down', color=color),
                    name='Entry'
                ),
                row=1, col=1
            )
            # Punkt wyjścia
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_time],
                    y=[trade.exit_price],
                    mode='markers',
                    marker=dict(size=10, symbol='x', color=color),
                    name='Exit'
                ),
                row=1, col=1
            )
            
        # Krzywa kapitału
        equity_curve = self._calculate_equity_curve()
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve,
                name='Equity'
            ),
            row=2, col=1
        )
        
        # Rozkład zysków/strat
        profits = [trade.profit for trade in self.trades]
        fig.add_trace(
            go.Histogram(
                x=profits,
                name='Trade P/L'
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=1200, title_text="Backtest Results Dashboard")
        return fig
        
    def _calculate_equity_curve(self) -> pd.Series:
        """Oblicza krzywą kapitału"""
        equity = pd.Series(index=self.data.index, data=10000.0)  # początkowy kapitał
        for trade in self.trades:
            mask = (equity.index > trade.exit_time)
            if any(mask):
                equity[mask] += trade.profit
        return equity
        
    def save_dashboard(self, filename: str = 'backtest_results.html'):
        """Zapisuje dashboard do pliku HTML"""
        fig = self.create_dashboard()
        fig.write_html(filename)

6. Zaktualizuj plik main.py o funkcję do uruchamiania backtestu:
async def run_backtest():
    logger = TradingLogger()
    config = Config.load_config()
    
    # Inicjalizacja strategii
    strategy = BasicStrategy(
        mt5_connector=None,  # Nie potrzebujemy połączenia w backteście
        ollama_connector=None,
        anthropic_connector=None,
        db_handler=None,
        config={
            'max_position_size': 0.1,
            'max_risk_per_trade': 0.02,
            'allowed_symbols': ['EURUSD']
        }
    )
    
    # Utworzenie backtestera
    backtester = Backtester(
        strategy=strategy,
        symbol='EURUSD',
        timeframe='1H',
        initial_capital=10000,
        start_date=datetime.now() - timedelta(days=30),
        logger=logger
    )
    
    # Uruchomienie backtestu
    results = await backtester.run_backtest()
    
    # Wizualizacja wyników
    visualizer = BacktestVisualizer(backtester.data, backtester.trades)
    visualizer.save_dashboard('backtest_results.html')
    
    print("\nWYNIKI BACKTESTU:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
