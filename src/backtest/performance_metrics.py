"""
Moduł do obliczania metryk wydajności strategii.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TradeResult:
    """Klasa reprezentująca wynik pojedynczej transakcji."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str  # 'BUY' lub 'SELL'
    profit: float
    size: float

class PerformanceMetrics:
    """Klasa do obliczania metryk wydajności strategii."""
    
    def __init__(self, trades: List[TradeResult], initial_capital: float = 10000):
        """
        Inicjalizacja kalkulatora metryk.
        
        Args:
            trades: Lista wykonanych transakcji
            initial_capital: Początkowy kapitał
        """
        self.trades = trades
        self.initial_capital = initial_capital
        
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Oblicza metryki wydajności strategii.
        
        Returns:
            Słownik z metrykami wydajności
        """
        if not self.trades:
            return {
                "total_return": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "total_trades": 0,
                "avg_profit": 0,
                "avg_win": 0,
                "avg_loss": 0
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
            "total_trades": total_trades,
            "avg_profit": np.mean([t.profit for t in self.trades]),
            "avg_win": np.mean([t.profit for t in self.trades if t.profit > 0]) if profitable_trades > 0 else 0,
            "avg_loss": np.mean([t.profit for t in self.trades if t.profit < 0]) if total_trades - profitable_trades > 0 else 0
        }
        
    def _calculate_equity_curve(self) -> np.ndarray:
        """
        Oblicza krzywą kapitału.
        
        Returns:
            Tablica numpy z wartościami kapitału
        """
        equity = [self.initial_capital]
        for trade in self.trades:
            equity.append(equity[-1] + trade.profit)
        return np.array(equity)
        
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Oblicza maksymalny drawdown.
        
        Args:
            equity_curve: Krzywa kapitału
            
        Returns:
            Wartość maksymalnego drawdownu (0-1)
        """
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _calculate_sharpe_ratio(self, equity_curve: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Oblicza wskaźnik Sharpe'a.
        
        Args:
            equity_curve: Krzywa kapitału
            risk_free_rate: Stopa wolna od ryzyka (roczna)
            
        Returns:
            Wartość wskaźnika Sharpe'a
        """
        returns = np.diff(equity_curve) / equity_curve[:-1]
        excess_returns = returns - risk_free_rate/252  # dzienny risk-free rate
        
        if len(excess_returns) < 2:
            return 0
            
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0 