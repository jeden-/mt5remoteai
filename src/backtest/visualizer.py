"""
Moduł do wizualizacji wyników backtestów.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict
from .performance_metrics import TradeResult

class BacktestVisualizer:
    """Klasa do wizualizacji wyników backtestów."""
    
    def __init__(self, data: pd.DataFrame, trades: List[TradeResult]):
        """
        Inicjalizacja wizualizatora.
        
        Args:
            data: DataFrame z danymi historycznymi
            trades: Lista wykonanych transakcji
        """
        self.data = data
        self.trades = trades
        
    def create_dashboard(self, show_indicators: bool = True) -> go.Figure:
        """
        Tworzy interaktywny dashboard z wynikami backtestu.
        
        Args:
            show_indicators: Czy wyświetlać wskaźniki techniczne (domyślnie True)
            
        Returns:
            Figura plotly z dashboardem
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Cena i Transakcje', 'Krzywa Kapitału', 'Rozkład Zysków/Strat'),
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Wykres świecowy
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['open'],
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close'],
                name='Cena'
            ),
            row=1, col=1
        )
        
        # Dodaj SMA jeśli show_indicators=True i kolumny istnieją
        if show_indicators:
            if 'SMA_20' in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data['SMA_20'],
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data['SMA_50'],
                        name='SMA 50',
                        line=dict(color='red', width=1)
                    ),
                    row=1, col=1
                )
        
        # Dodaj punkty wejścia/wyjścia
        for trade in self.trades:
            if trade.entry_time in self.data.index and trade.exit_time in self.data.index:
                color = 'green' if trade.profit > 0 else 'red'
                # Punkt wejścia
                fig.add_trace(
                    go.Scatter(
                        x=[trade.entry_time],
                        y=[trade.entry_price],
                        mode='markers',
                        marker=dict(
                            size=10, 
                            symbol='triangle-up' if trade.direction == 'BUY' else 'triangle-down',
                            color=color
                        ),
                        name='Wejścia'
                    ),
                    row=1, col=1
                )
                # Punkt wyjścia
                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[trade.exit_price],
                        mode='markers',
                        marker=dict(
                            size=10,
                            symbol='x',
                            color=color
                        ),
                        name='Wyjścia'
                    ),
                    row=1, col=1
                )
            
        # Krzywa kapitału
        equity_curve = self._calculate_equity_curve()
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve,
                name='Kapitał',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # Rozkład zysków/strat
        profits = [trade.profit for trade in self.trades]
        fig.add_trace(
            go.Histogram(
                x=profits,
                name='Rozkład P/L',
                nbinsx=30,
                marker_color='blue'
            ),
            row=3, col=1
        )
        
        # Aktualizuj układ
        fig.update_layout(
            height=1200,
            title_text="Dashboard Wyników Backtestu",
            showlegend=True
        )
        
        # Aktualizuj osie
        fig.update_xaxes(title_text="Data", row=3, col=1)
        fig.update_yaxes(title_text="Cena", row=1, col=1)
        fig.update_yaxes(title_text="Kapitał", row=2, col=1)
        fig.update_yaxes(title_text="Liczba transakcji", row=3, col=1)
        
        # Wyłącz rangeslider
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
        
    def _calculate_equity_curve(self) -> pd.Series:
        """
        Oblicza krzywą kapitału.
        
        Returns:
            Series z wartościami kapitału
        """
        equity = pd.Series(index=self.data.index, data=10000.0)  # początkowy kapitał
        for trade in self.trades:
            if trade.exit_time in equity.index:
                mask = (equity.index > trade.exit_time)
                if any(mask):
                    equity[mask] += trade.profit
        return equity
        
    def save_dashboard(self, filename: str = 'backtest_results.html'):
        """
        Zapisuje dashboard do pliku HTML.
        
        Args:
            filename: Nazwa pliku do zapisu
        """
        fig = self.create_dashboard()
        fig.write_html(filename)
        print(f"🥷 Dashboard zapisany do {filename}") 