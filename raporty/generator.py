"""
Moduł do generowania raportów z wyników backtestingu.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

def generuj_wykresy(df: pd.DataFrame, 
                    sciezka_wyjsciowa: Path) -> Dict[str, str]:
    """
    Generuje wykresy dla raportu backtestingu.
    
    Args:
        df: DataFrame z danymi
        sciezka_wyjsciowa: Ścieżka do katalogu wyjściowego
        
    Returns:
        Słownik z nazwami wygenerowanych plików
    """
    try:
        pliki = {}
        
        # Wykres kapitału
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['kapital'],
                               mode='lines',
                               name='Kapitał'))
        fig.update_layout(title='Zmiana kapitału w czasie',
                         xaxis_title='Data',
                         yaxis_title='Kapitał')
        sciezka = sciezka_wyjsciowa / 'kapital.html'
        fig.write_html(str(sciezka))
        pliki['kapital'] = str(sciezka)
        
        # Rozkład zwrotów
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['zwrot'],
                                 nbinsx=50,
                                 name='Rozkład zwrotów'))
        fig.update_layout(title='Rozkład zwrotów z transakcji',
                         xaxis_title='Zwrot %',
                         yaxis_title='Liczba transakcji')
        sciezka = sciezka_wyjsciowa / 'zwroty.html'
        fig.write_html(str(sciezka))
        pliki['zwroty'] = str(sciezka)
        
        # Wykres drawdown
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'],
                               mode='lines',
                               name='Drawdown'))
        fig.update_layout(title='Drawdown w czasie',
                         xaxis_title='Data',
                         yaxis_title='Drawdown %')
        sciezka = sciezka_wyjsciowa / 'drawdown.html'
        fig.write_html(str(sciezka))
        pliki['drawdown'] = str(sciezka)
        
        return pliki
        
    except Exception as e:
        logger.error("❌ Błąd podczas generowania wykresów: %s", str(e))
        return {}

def generuj_raport_html(df: pd.DataFrame,
                       statystyki: Dict[str, Any],
                       sciezka_wyjsciowa: Path) -> str:
    """
    Generuje raport HTML z wynikami backtestingu.
    
    Args:
        df: DataFrame z danymi
        statystyki: Słownik ze statystykami
        sciezka_wyjsciowa: Ścieżka do katalogu wyjściowego
        
    Returns:
        Ścieżka do wygenerowanego pliku HTML
    """
    try:
        # Generujemy wykresy
        wykresy = generuj_wykresy(df, sciezka_wyjsciowa)
        
        # Tworzymy szablon HTML
        html = f"""
        <html>
        <head>
            <title>Raport backtestingu - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                .chart {{ width: 100%; height: 400px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Raport backtestingu</h1>
            <p>Data wygenerowania: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <h2>Statystyki</h2>
            <table>
                <tr>
                    <th>Metryka</th>
                    <th>Wartość</th>
                </tr>
        """
        
        # Dodajemy statystyki
        for nazwa, wartosc in statystyki.items():
            if isinstance(wartosc, float):
                wartosc = f"{wartosc:.2f}"
            html += f"""
                <tr>
                    <td>{nazwa}</td>
                    <td>{wartosc}</td>
                </tr>
            """
            
        html += """
            </table>
            
            <h2>Wykresy</h2>
        """
        
        # Dodajemy wykresy
        for nazwa, sciezka in wykresy.items():
            html += f"""
            <h3>{nazwa.title()}</h3>
            <iframe class="chart" src="{sciezka}"></iframe>
            """
            
        html += """
        </body>
        </html>
        """
        
        # Zapisujemy raport
        sciezka = sciezka_wyjsciowa / 'raport.html'
        with open(sciezka, 'w', encoding='utf-8') as f:
            f.write(html)
            
        return str(sciezka)
        
    except Exception as e:
        logger.error("❌ Błąd podczas generowania raportu HTML: %s", str(e))
        return "" 