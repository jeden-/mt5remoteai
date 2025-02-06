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
    
    VALID_TIMEFRAMES = ['1M', '5M', '15M', '30M', '1H', '4H', '1D', '1W', '1MN']
    
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
            
        Raises:
            ValueError: Gdy parametry są nieprawidłowe
        """
        # Walidacja timeframe
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Nieprawidłowy timeframe. Dozwolone wartości: {', '.join(self.VALID_TIMEFRAMES)}")
            
        # Walidacja initial_capital
        if initial_capital <= 0:
            raise ValueError("Initial capital musi być większy od 0")
            
        # Walidacja start_date
        if start_date and start_date > datetime.now():
            raise ValueError("Data początkowa nie może być z przyszłości")
            
        # Walidacja symbol
        if not symbol or len(symbol) < 3 or len(symbol) > 10:
            raise ValueError("Symbol musi mieć od 3 do 10 znaków")
            
        # Walidacja strategy
        if not isinstance(strategy, BasicStrategy):
            raise ValueError("Strategia musi być instancją BasicStrategy")
        
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
        
        # Załaduj dane z MT5 jeśli nie są ustawione
        if self.data is None:
            data_loader = HistoricalDataLoader(self.symbol, self.timeframe, self.start_date)
            self.data = await data_loader.load_data()
        
        # Sprawdź czy dane są poprawne
        if self.data is None or len(self.data) == 0:
            raise RuntimeError(f"❌ Nie udało się załadować danych dla {self.symbol}")
            
        print("\n🔍 Pierwsze 5 świec po załadowaniu danych:")
        for i in range(min(5, len(self.data))):
            print(f"Świeca {i}:")
            print(f"  Open: {self.data['open'].iloc[i]}")
            print(f"  High: {self.data['high'].iloc[i]}")
            print(f"  Low: {self.data['low'].iloc[i]}")
            print(f"  Close: {self.data['close'].iloc[i]}")
            
        current_position = None
        
        for i in range(len(self.data) - 1):  # Iterujemy do przedostatniej świecy
            current_bar = self.data.iloc[i]
            next_bar = self.data.iloc[i + 1]
            
            # Przygotuj dane dla strategii
            market_data = {
                'symbol': self.symbol,
                'current_price': float(current_bar['close']),
                'next_open': float(next_bar['open']),
                'technical_indicators': {
                    'sma_20': float(current_bar['SMA_20']),
                    'sma_50': float(current_bar['SMA_50']),
                    'rsi': float(current_bar['RSI']),
                    'macd': float(current_bar['MACD']),
                    'signal_line': float(current_bar['Signal_Line'])
                },
                'position': current_position,
                'analysis_summary': {
                    'market_data': {
                        'volume': 0.1,
                        'entry_price': float(current_bar['close']),
                        'stop_loss': float(current_bar['close']) * 0.995,
                        'take_profit': float(current_bar['close']) * 1.01
                    },
                    'ai_recommendations': {}
                }
            }
            
            try:
                # Generuj sygnały
                signals = await self.strategy.generate_signals(market_data)
                action = signals['market_data']['action']
                
                if i < 5:  # Loguj tylko pierwsze 5 iteracji
                    print(f"\n🔍 Iteracja {i}:")
                    print(f"  Current bar:")
                    print(f"    Open: {current_bar['open']}")
                    print(f"    High: {current_bar['high']}")
                    print(f"    Low: {current_bar['low']}")
                    print(f"    Close: {current_bar['close']}")
                    print(f"  Next bar:")
                    print(f"    Open: {next_bar['open']}")
                    print(f"    High: {next_bar['high']}")
                    print(f"    Low: {next_bar['low']}")
                    print(f"    Close: {next_bar['close']}")
                    print(f"  Signal:")
                    print(f"    Action: {action}")
                    print(f"    Entry price: {signals['market_data'].get('entry_price')}")
                
                self.logger.log_trade({
                    'type': 'DEBUG',
                    'symbol': self.symbol,
                    'message': f"🔍 Otrzymany sygnał: {action}, cena: {market_data['current_price']}, next_open: {market_data['next_open']}"
                })
                
                # Sprawdź warunki zamknięcia dla otwartej pozycji
                if current_position is not None:
                    should_close, exit_price = self._check_close_conditions(current_position, next_bar, signals)
                    if should_close:
                        trade = current_position['trade']
                        trade.exit_time = next_bar.name
                        trade.exit_price = exit_price
                        trade.profit = self._calculate_profit(current_position, exit_price)
                        self.trades.append(trade)
                        
                        self.logger.log_trade({
                            'type': 'CLOSE',
                            'symbol': self.symbol,
                            'volume': trade.size,
                            'price': exit_price,
                            'profit': trade.profit,
                            'message': f"🥷 Zamykam pozycję {trade.direction} na {self.symbol} po cenie {exit_price}, profit: {trade.profit}"
                        })
                        
                        current_position = None
                
                # Jeśli nie mamy pozycji i otrzymaliśmy sygnał otwarcia
                if current_position is None and action in ['BUY', 'SELL']:
                    # Sprawdź czy cena otwarcia następnej świecy nie jest zbyt daleko od ceny z sygnału
                    signal_price = float(signals['market_data'].get('entry_price', current_bar['close']))
                    entry_price = float(next_bar['open'])
                    
                    print(f"\n🔍 Sprawdzam warunki wejścia:")
                    print(f"  Action: {action}")
                    print(f"  Signal price: {signal_price}")
                    print(f"  Entry price: {entry_price}")
                    print(f"  High: {current_bar['high']}")
                    print(f"  Low: {current_bar['low']}")
                    
                    # Dla BUY akceptujemy każdą cenę niższą od sygnału
                    # Dla cen wyższych sprawdzamy czy nie są zbyt wysokie
                    if action == 'BUY':
                        if entry_price <= signal_price:
                            # Akceptujemy każdą cenę niższą od sygnału dla BUY
                            print(f"✅ Akceptuję BUY - cena wejścia niższa od sygnału: {entry_price} <= {signal_price}")
                            pass
                        elif entry_price > current_bar['high'] * 1.01:  # Max 1% powyżej high
                            print(f"⚠️ Cena otwarcia zbyt wysoka dla BUY: {entry_price} vs {current_bar['high']}")
                            continue
                        else:
                            print(f"✅ Akceptuję BUY - cena wejścia w dopuszczalnym zakresie: {entry_price} <= {current_bar['high'] * 1.01}")
                    else:  # SELL
                        if entry_price >= signal_price:
                            # Akceptujemy każdą cenę wyższą od sygnału dla SELL
                            print(f"✅ Akceptuję SELL - cena wejścia wyższa od sygnału: {entry_price} >= {signal_price}")
                            pass
                        elif entry_price < current_bar['low'] * 0.99:  # Max 1% poniżej low
                            print(f"⚠️ Cena otwarcia zbyt niska dla SELL: {entry_price} vs {current_bar['low']}")
                            continue
                        else:
                            print(f"✅ Akceptuję SELL - cena wejścia w dopuszczalnym zakresie: {entry_price} >= {current_bar['low'] * 0.99}")
                    
                    # Użyj SL/TP z sygnału lub oblicz na podstawie ceny wejścia
                    if 'stop_loss' in signals['market_data'] and 'take_profit' in signals['market_data']:
                        stop_loss = float(signals['market_data']['stop_loss'])
                        take_profit = float(signals['market_data']['take_profit'])
                        
                        # Dostosuj SL/TP do ceny wejścia
                        if action == 'BUY':
                            sl_distance = (signal_price - stop_loss) / signal_price
                            tp_distance = (take_profit - signal_price) / signal_price
                            stop_loss = entry_price * (1 - sl_distance)
                            take_profit = entry_price * (1 + tp_distance)
                        else:  # SELL
                            sl_distance = (stop_loss - signal_price) / signal_price
                            tp_distance = (signal_price - take_profit) / signal_price
                            stop_loss = entry_price * (1 + sl_distance)
                            take_profit = entry_price * (1 - tp_distance)
                    else:
                        if action == 'BUY':
                            stop_loss = entry_price * 0.995
                            take_profit = entry_price * 1.01
                        else:  # SELL
                            stop_loss = entry_price * 1.005
                            take_profit = entry_price * 0.99
                    
                    # Sprawdź czy SL/TP są prawidłowe
                    if action == 'BUY':
                        if stop_loss >= entry_price or take_profit <= entry_price:
                            self.logger.log_trade({
                                'type': 'WARNING',
                                'symbol': self.symbol,
                                'message': f"⚠️ Nieprawidłowe poziomy SL/TP dla BUY: SL={stop_loss}, TP={take_profit}, entry={entry_price}"
                            })
                            continue
                    else:  # SELL
                        if stop_loss <= entry_price or take_profit >= entry_price:
                            self.logger.log_trade({
                                'type': 'WARNING',
                                'symbol': self.symbol,
                                'message': f"⚠️ Nieprawidłowe poziomy SL/TP dla SELL: SL={stop_loss}, TP={take_profit}, entry={entry_price}"
                            })
                            continue
                    
                    trade = TradeResult(
                        entry_time=next_bar.name,
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        direction=action,
                        profit=None,
                        size=float(signals['market_data']['volume'])
                    )
                    
                    current_position = {
                        'entry_time': next_bar.name,
                        'entry_price': entry_price,
                        'direction': action,
                        'size': float(signals['market_data']['volume']),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'trade': trade
                    }
                    
                    self.logger.log_trade({
                        'type': action,
                        'symbol': self.symbol,
                        'volume': current_position['size'],
                        'price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'message': f"🥷 Otwieram pozycję {action} na {self.symbol} po cenie {entry_price}, SL: {stop_loss}, TP: {take_profit}"
                    })
                
            except Exception as e:
                self.logger.log_error(f"❌ Błąd podczas backtestingu: {str(e)}")
                if current_position is not None:
                    trade = current_position['trade']
                    trade.exit_time = next_bar.name
                    trade.exit_price = float(next_bar['open'])
                    trade.profit = self._calculate_profit(current_position, trade.exit_price)
                    self.trades.append(trade)
                    current_position = None
                    self.logger.log_error(f"❌ Zamykam pozycję z powodu błędu na {self.symbol}")
                
        # Zamknij otwarte pozycje na końcu backtestingu
        if current_position is not None:
            last_bar = self.data.iloc[-1]
            trade = current_position['trade']
            trade.exit_time = last_bar.name
            trade.exit_price = float(last_bar['close'])
            trade.profit = self._calculate_profit(current_position, trade.exit_price)
            self.trades.append(trade)
            
            self.logger.log_trade({
                'type': 'CLOSE',
                'symbol': self.symbol,
                'volume': trade.size,
                'price': trade.exit_price,
                'profit': trade.profit,
                'message': f"🥷 Zamykam pozycję na końcu backtestingu {trade.direction} na {self.symbol} po cenie {trade.exit_price}, profit: {trade.profit}"
            })
                        
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
        if signals['market_data']['action'] == 'CLOSE':
            return True, float(current_bar['open'])
            
        # Sprawdź przeciwny sygnał (dla BUY - SELL, dla SELL - BUY)
        if position['direction'] == 'BUY' and signals['market_data']['action'] == 'SELL':
            return True, float(current_bar['open'])
        elif position['direction'] == 'SELL' and signals['market_data']['action'] == 'BUY':
            return True, float(current_bar['open'])
            
        # Sprawdź gap down/up na otwarciu
        if position['direction'] == 'BUY':
            if current_bar['open'] < position['stop_loss']:  # Gap down
                self.logger.log_trade({
                    'type': 'CLOSE',
                    'symbol': self.symbol,
                    'message': f"🛑 Stop Loss hit (gap) dla pozycji BUY na {self.symbol} po cenie {current_bar['open']}, open: {current_bar['open']}"
                })
                return True, current_bar['open']
            elif current_bar['open'] > position['take_profit']:  # Gap up
                self.logger.log_trade({
                    'type': 'CLOSE',
                    'symbol': self.symbol,
                    'message': f"🎯 Take Profit hit (gap) dla pozycji BUY na {self.symbol} po cenie {current_bar['open']}, open: {current_bar['open']}"
                })
                return True, current_bar['open']
        else:  # SELL
            if current_bar['open'] > position['stop_loss']:  # Gap up
                self.logger.log_trade({
                    'type': 'CLOSE',
                    'symbol': self.symbol,
                    'message': f"🛑 Stop Loss hit (gap) dla pozycji SELL na {self.symbol} po cenie {current_bar['open']}, open: {current_bar['open']}"
                })
                return True, current_bar['open']
            elif current_bar['open'] < position['take_profit']:  # Gap down
                self.logger.log_trade({
                    'type': 'CLOSE',
                    'symbol': self.symbol,
                    'message': f"🎯 Take Profit hit (gap) dla pozycji SELL na {self.symbol} po cenie {current_bar['open']}, open: {current_bar['open']}"
                })
                return True, current_bar['open']
            
        # Sprawdź stop loss
        if position['direction'] == 'BUY':
            if current_bar['low'] <= position['stop_loss']:
                self.logger.log_trade({
                    'type': 'CLOSE',
                    'symbol': self.symbol,
                    'message': f"🛑 Stop Loss hit dla pozycji BUY na {self.symbol} po cenie {position['stop_loss']}, low: {current_bar['low']}"
                })
                return True, position['stop_loss']
        else:  # SELL
            if current_bar['high'] >= position['stop_loss']:
                self.logger.log_trade({
                    'type': 'CLOSE',
                    'symbol': self.symbol,
                    'message': f"🛑 Stop Loss hit dla pozycji SELL na {self.symbol} po cenie {position['stop_loss']}, high: {current_bar['high']}"
                })
                return True, position['stop_loss']
                
        # Sprawdź take profit
        if position['direction'] == 'BUY':
            if current_bar['high'] >= position['take_profit']:
                self.logger.log_trade({
                    'type': 'CLOSE',
                    'symbol': self.symbol,
                    'message': f"🎯 Take Profit hit dla pozycji BUY na {self.symbol} po cenie {position['take_profit']}, high: {current_bar['high']}"
                })
                return True, position['take_profit']
        else:  # SELL
            if current_bar['low'] <= position['take_profit']:
                self.logger.log_trade({
                    'type': 'CLOSE',
                    'symbol': self.symbol,
                    'message': f"🎯 Take Profit hit dla pozycji SELL na {self.symbol} po cenie {position['take_profit']}, low: {current_bar['low']}"
                })
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