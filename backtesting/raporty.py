"""
Moduł do generowania raportów z wyników backtestingu w systemie NikkeiNinja.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from backtesting.metryki import generuj_raport_dzienny, oblicz_metryki_ryzyka
from backtesting.symulator import WynikBacktestu
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def generuj_wykresy(
    wynik: WynikBacktestu,
    dane: pd.DataFrame,
    sciezka: Path
) -> Dict[str, Any]:
    """Generuje wykresy dla raportu backtestingu.
    
    Args:
        wynik: Obiekt z wynikami backtestingu
        dane: DataFrame z danymi historycznymi
        sciezka: Ścieżka bazowa do zapisu wykresów
        
    Returns:
        Słownik z nazwami wygenerowanych plików
    """
    try:
        # Wykres kapitału
        kapitaly = []
        kapital = 100000  # Początkowy kapitał
        for t in wynik.transakcje:
            kapital += t.zysk
            kapitaly.append(kapital)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[t.data_zamkniecia for t in wynik.transakcje],
            y=kapitaly,
            mode='lines',
            name='Kapitał'
        ))
        fig.update_layout(title='Krzywa kapitału')
        fig.write_image(str(sciezka.parent / 'kapital.png'))
        
        # Wykres rozkładu zwrotów
        zwroty = pd.Series([t.zysk for t in wynik.transakcje]).pct_change()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=zwroty, nbinsx=20))
        fig.update_layout(title='Rozkład zwrotów')
        fig.write_image(str(sciezka.parent / 'zwroty.png'))
        
        # Wykres drawdown
        drawdowns = []
        max_kapital = 100000
        for k in kapitaly:
            max_kapital = max(max_kapital, k)
            drawdown = (max_kapital - k) / max_kapital
            drawdowns.append(drawdown)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[t.data_zamkniecia for t in wynik.transakcje],
            y=drawdowns,
            mode='lines',
            name='Drawdown'
        ))
        fig.update_layout(title='Drawdown')
        fig.write_image(str(sciezka.parent / 'drawdown.png'))
        
        return {
            'kapital': 'kapital.png',
            'zwroty': 'zwroty.png',
            'drawdown': 'drawdown.png'
        }
        
    except Exception as e:
        logger.error(f"❌ Błąd podczas generowania wykresów: {str(e)}")
        return {}


def generuj_raport_html(
    wynik: WynikBacktestu,
    dane: pd.DataFrame,
    sciezka: Path
) -> None:
    """Generuje raport HTML z wynikami backtestingu.
    
    Args:
        wynik: Obiekt z wynikami backtestingu
        dane: DataFrame z danymi historycznymi
        sciezka: Ścieżka do zapisu raportu
    """
    try:
        # Generowanie wykresów
        wykresy = generuj_wykresy(wynik, dane, sciezka)
        
        # Przygotowanie danych do raportu
        statystyki = {
            'Nazwa strategii': wynik.nazwa_strategii,
            'Data rozpoczęcia': wynik.data_rozpoczecia.strftime('%Y-%m-%d %H:%M'),
            'Data zakończenia': wynik.data_zakonczenia.strftime('%Y-%m-%d %H:%M'),
            'Liczba transakcji': wynik.liczba_transakcji,
            'Zysk całkowity': f"{wynik.zysk_calkowity:.2f}",
            'Zysk procentowy': f"{wynik.zysk_procent:.2f}%",
            'Win rate': f"{wynik.win_rate*100:.1f}%",
            'Profit factor': f"{wynik.profit_factor:.2f}",
            'Max drawdown': f"{wynik.max_drawdown*100:.1f}%",
            'Sharpe ratio': f"{wynik.sharpe_ratio:.2f}"
        }
        
        # Generowanie HTML
        html = """
        <html>
        <head>
            <title>Raport backtestingu</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f5f5f5; }
                img { max-width: 100%; margin: 10px 0; }
                .section { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Raport backtestingu</h1>
            
            <div class="section">
                <h2>Statystyki</h2>
                <table>
                    <tr><th>Metryka</th><th>Wartość</th></tr>
        """
        
        for nazwa, wartosc in statystyki.items():
            html += f"<tr><td>{nazwa}</td><td>{wartosc}</td></tr>"
            
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Wykresy</h2>
        """
        
        for nazwa, plik in wykresy.items():
            html += f'<img src="{plik}" alt="{nazwa}"><br>'
            
        html += """
            </div>
            
            <div class="section">
                <h2>Lista transakcji</h2>
                <table>
                    <tr>
                        <th>Data otwarcia</th>
                        <th>Data zamknięcia</th>
                        <th>Kierunek</th>
                        <th>Cena otwarcia</th>
                        <th>Cena zamknięcia</th>
                        <th>Wolumen</th>
                        <th>Prowizja</th>
                        <th>Zysk</th>
                    </tr>
        """
        
        for t in wynik.transakcje:
            html += f"""
                <tr>
                    <td>{t.data_otwarcia.strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{t.data_zamkniecia.strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{t.kierunek}</td>
                    <td>{t.cena_otwarcia:.2f}</td>
                    <td>{t.cena_zamkniecia:.2f}</td>
                    <td>{t.wolumen:.2f}</td>
                    <td>{t.prowizja:.2f}</td>
                    <td>{t.zysk:.2f}</td>
                </tr>
            """
            
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Zapis raportu
        sciezka.parent.mkdir(parents=True, exist_ok=True)
        sciezka.write_text(html, encoding='utf-8')
        logger.info(f"🥷 Wygenerowano raport HTML: {sciezka}")
        
    except Exception as e:
        logger.error(f"❌ Błąd podczas generowania raportu HTML: {str(e)}")
        raise


def generuj_raport_json(wynik: WynikBacktestu,
                       sciezka_wyjsciowa: str) -> None:
    """
    Generuje raport w formacie JSON z wynikami backtestingu.
    
    Przydatne do:
    - Dalszej analizy wyników
    - Porównywania strategii
    - Optymalizacji parametrów
    """
    try:
        # Przygotowanie danych
        df_dzienny = generuj_raport_dzienny(wynik.transakcje)
        if df_dzienny.empty:
            logger.warning("⚠️ Brak danych do wygenerowania raportu JSON")
            return
            
        # Obliczanie metryk
        historia_kapitalu = df_dzienny['skumulowany_wynik'].tolist()
        metryki = oblicz_metryki_ryzyka(wynik.transakcje, historia_kapitalu)
        
        # Przygotowanie danych do eksportu
        dane = {
            'nazwa_strategii': wynik.nazwa_strategii,
            'data_rozpoczecia': wynik.data_rozpoczecia.isoformat(),
            'data_zakonczenia': wynik.data_zakonczenia.isoformat(),
            'metryki': metryki,
            'statystyki_dzienne': df_dzienny.to_dict(orient='records'),
            'transakcje': [
                {
                    'timestamp_wejscia': t.timestamp_wejscia.isoformat(),
                    'timestamp_wyjscia': t.timestamp_wyjscia.isoformat(),
                    'kierunek': t.kierunek.value,
                    'cena_wejscia': float(t.cena_wejscia),
                    'cena_wyjscia': float(t.cena_wyjscia),
                    'wolumen': float(t.wolumen),
                    'zysk_procent': t.zysk_procent,
                    'metadane': t.metadane
                }
                for t in wynik.transakcje
            ]
        }
        
        # Zapisanie raportu
        sciezka = Path(sciezka_wyjsciowa) / f'raport_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        pd.DataFrame(dane).to_json(sciezka, orient='records', lines=True)
        
        logger.info("🥷 Wygenerowano raport JSON: %s", sciezka)
        
    except Exception as e:
        logger.error("❌ Błąd podczas generowania raportu JSON: %s", str(e)) 