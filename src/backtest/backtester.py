"""
Moduł do przeprowadzania backtestów strategii tradingowych.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from .data_loader import HistoricalDataLoader
from .performance_metrics import PerformanceMetrics, TradeResult
from ..strategies.basic_strategy import BasicStrategy
from ..utils.logger import TradingLogger

class Backtester:
    """Klasa do przeprowadzania backtestów strategii tradingowych."""
    
    def __init__(
        self,
        strategy: BasicStrategy,
        symbol: str,
        timeframe: str = "1H",
        initial_capital: float = 10000,
        start_date: Optional[datetime] = None,
        logger: Optional[TradingLogger] = None
    ):
        """
        Inicjalizacja backtestera.
        
        Args:
            strategy: Strategia do przetestowania
            symbol: Symbol instrumentu
            timeframe: Interwał czasowy
            initial_capital: Początkowy kapitał
            start_date: Data początkowa
            logger: Logger do zapisywania operacji
        """
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.logger = logger or TradingLogger(strategy_name=strategy.name)
        self.trades: List[TradeResult] = []
        self.data: Optional[pd.DataFrame] = None
        
    async def run_backtest(self) -> Dict[str, float]:
        """
        Przeprowadza backtesting strategii.
        
        Returns:
            Słownik z metrykami wydajności
            
        Raises:
            RuntimeError: Gdy nie uda się załadować danych
        """
        self.logger.log_trade({
            'type': 'INFO',
            'symbol': self.symbol,
            'message': f"🥷 Rozpoczynam backtest dla {self.symbol}"
        })
        
        # Załaduj dane historyczne
        data_loader = HistoricalDataLoader(self.symbol, self.timeframe, self.start_date)
        self.data = await data_loader.load_data()
        
        if self.data is None or len(self.data) == 0:
            raise RuntimeError(f"❌ Nie udało się załadować danych dla {self.symbol}")
        
        current_position = None
        
        for i in range(len(self.data)):
            current_data = self.data.iloc[:i+1]
            next_bar = self.data.iloc[i] if i == len(self.data)-1 else self.data.iloc[i+1]
            
            # Przygotuj dane dla strategii
            market_data = {
                'symbol': self.symbol,
                'current_price': float(current_data['close'].iloc[-1]),
                'sma_20': float(current_data['SMA_20'].iloc[-1]),
                'sma_50': float(current_data['SMA_50'].iloc[-1]),
                'rsi': float(current_data['RSI'].iloc[-1]),
                'macd': float(current_data['MACD'].iloc[-1]),
                'signal_line': float(current_data['Signal_Line'].iloc[-1])
            }
            
            try:
                # Generuj sygnały
                signals = await self.strategy.generate_signals({'market_data': market_data})
                
                # Wykonaj transakcje
                if current_position is None:
                    if signals['action'] in ['BUY', 'SELL']:
                        # Otwórz pozycję
                        current_position = {
                            'entry_time': current_data.index[-1],
                            'entry_price': next_bar['open'],
                            'direction': signals['action'],
                            'size': signals.get('volume', self.strategy.config['max_position_size']),
                            'stop_loss': signals.get('stop_loss'),
                            'take_profit': signals.get('take_profit')
                        }
                        self.logger.log_trade({
                            'type': 'DEBUG',
                            'symbol': self.symbol,
                            'message': f"🥷 Otwieram pozycję {signals['action']} na {self.symbol}"
                        })
                else:
                    # Sprawdź warunki zamknięcia
                    should_close, exit_price = self._check_close_conditions(current_position, next_bar, signals)
                    if should_close:
                        # Zamknij pozycję
                        trade = TradeResult(
                            entry_time=current_position['entry_time'],
                            exit_time=next_bar.name,
                            entry_price=current_position['entry_price'],
                            exit_price=exit_price,
                            direction=current_position['direction'],
                            profit=self._calculate_profit(current_position, exit_price),
                            size=current_position['size']
                        )
                        self.trades.append(trade)
                        
                        self.logger.log_trade({
                            'type': trade.direction,
                            'symbol': self.symbol,
                            'volume': trade.size,
                            'price': trade.exit_price,
                            'profit': trade.profit
                        })
                            
                        current_position = None
                        self.logger.log_trade({
                            'type': 'DEBUG',
                            'symbol': self.symbol,
                            'message': f"🥷 Zamykam pozycję na {self.symbol}"
                        })
                            
            except Exception as e:
                self.logger.log_error(f"❌ Błąd podczas backtestingu: {str(e)}")
                if current_position is not None:
                    # W przypadku błędu zamykamy pozycję
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
                    self.logger.log_error(f"❌ Zamykam pozycję z powodu błędu na {self.symbol}")
                continue
                        
        # Oblicz metryki wydajności
        metrics = PerformanceMetrics(self.trades, self.initial_capital)
        results = metrics.calculate_metrics()
        
        self.logger.log_trade({
            'type': 'INFO',
            'symbol': self.symbol,
            'message': f"🥷 Backtest zakończony dla {self.symbol}"
        })
        self.logger.log_trade({
            'type': 'INFO',
            'symbol': self.symbol,
            'message': f"📊 Wyniki: {results}"
        })
        
        return results
        
    def _check_close_conditions(self, position: Dict, current_bar: pd.Series, signals: Dict) -> Tuple[bool, float]:
        """
        Sprawdza warunki zamknięcia pozycji i zwraca odpowiednią cenę wyjścia.
        
        Args:
            position: Aktualna pozycja
            current_bar: Aktualna świeca
            signals: Sygnały ze strategii
            
        Returns:
            Tuple (czy_zamknąć, cena_wyjścia)
        """
        # Sprawdź sygnał zamknięcia ze strategii
        if signals['action'] == 'CLOSE':
            return True, current_bar['open']
            
        # Sprawdź stop loss
        if position['direction'] == 'BUY':
            if position['stop_loss'] and current_bar['low'] <= position['stop_loss']:
                return True, position['stop_loss']
            if position['take_profit'] and current_bar['high'] >= position['take_profit']:
                return True, position['take_profit']
        else:  # SELL
            if position['stop_loss'] and current_bar['high'] >= position['stop_loss']:
                return True, position['stop_loss']
            if position['take_profit'] and current_bar['low'] <= position['take_profit']:
                return True, position['take_profit']
                
        return False, 0.0
            
    def _calculate_profit(self, position: Dict, exit_price: float) -> float:
        """
        Oblicza zysk/stratę z transakcji.
        
        Args:
            position: Pozycja
            exit_price: Cena wyjścia
            
        Returns:
            Zysk/strata z transakcji
        """
        if position['direction'] == 'BUY':
            return (exit_price - position['entry_price']) * position['size']
        else:  # SELL
            return (position['entry_price'] - exit_price) * position['size'] 