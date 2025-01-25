"""
Testy dla strategii hybrydowej.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strategie.hybrydowa import StrategiaHybrydowa
from strategie.interfejs import KierunekTransakcji


@pytest.fixture
def dane_testowe():
    """Tworzy przykładowe dane OHLCV do testów."""
    daty = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    dane = pd.DataFrame({
        'open': np.random.normal(100, 2, 100),
        'high': np.random.normal(102, 2, 100),
        'low': np.random.normal(98, 2, 100),
        'close': np.random.normal(101, 2, 100),
        'volume': np.random.normal(1000, 200, 100)
    }, index=daty)
    
    # Korekta high/low
    dane['high'] = np.maximum(dane[['open', 'close', 'high']].max(axis=1), dane['high'])
    dane['low'] = np.minimum(dane[['open', 'close', 'low']].min(axis=1), dane['low'])
    
    return dane


@pytest.fixture
def strategia():
    """Tworzy instancję strategii z domyślnymi parametrami."""
    strategia = StrategiaHybrydowa()
    strategia.inicjalizuj({})
    return strategia


@pytest.mark.asyncio
async def test_inicjalizacja(strategia):
    """Test inicjalizacji strategii."""
    assert strategia.parametry['waga_techniczna'] == 0.4
    assert strategia.parametry['waga_wyckoff'] == 0.4
    assert strategia.parametry['waga_sentyment'] == 0.2
    assert strategia.parametry['min_pewnosc_sentymentu'] == 0.6
    assert strategia.parametry['min_podobienstwo_wzorca'] == 0.7


@pytest.mark.asyncio
async def test_analizuj_sentyment(strategia, dane_testowe):
    """Test analizy sentymentu."""
    # Mock dla RAG
    mock_sentyment = {
        'sredni_sentyment': 0.5,
        'zmiana_sentymentu': 0.3,
        'liczba_wzmianek': 10
    }
    mock_wzorce = [
        {'metadata': {'zmiana_proc': 2.5}},
        {'metadata': {'zmiana_proc': 1.5}},
        {'metadata': {'zmiana_proc': 3.0}}
    ]
    
    with patch.object(strategia.rag, 'agreguj_sentyment', return_value=mock_sentyment), \
         patch.object(strategia.rag, 'znajdz_podobne_wzorce', return_value=mock_wzorce):
        
        wynik = await strategia._analizuj_sentyment(dane_testowe)
        
        assert wynik['sentyment'] == 0.5
        assert wynik['pewnosc'] == 0.3
        assert wynik['liczba_wzmianek'] == 10
        assert wynik['srednia_zmiana'] == pytest.approx(2.33, rel=0.01)


@pytest.mark.asyncio
async def test_analizuj_sygnaly(strategia, dane_testowe):
    """Test generowania sygnałów."""
    # Mock dla strategii składowych
    mock_tech_sygnal = Mock(
        kierunek=KierunekTransakcji.LONG,
        metadane={}
    )
    mock_wyckoff_sygnal = Mock(
        kierunek=KierunekTransakcji.SHORT,
        metadane={}
    )
    
    with patch.object(strategia.techniczna, 'analizuj', return_value=[mock_tech_sygnal]), \
         patch.object(strategia.wyckoff, 'analizuj', return_value=[mock_wyckoff_sygnal]), \
         patch.object(strategia, '_analizuj_sentyment', return_value={
             'sentyment': 0.5,
             'pewnosc': 0.8,
             'srednia_zmiana': 2.5,
             'liczba_wzmianek': 10
         }):
        
        sygnaly = await strategia.analizuj(dane_testowe)
        
        assert len(sygnaly) == 1  # Tylko sygnał z wysoką wagą
        assert sygnaly[0].kierunek == KierunekTransakcji.LONG
        assert sygnaly[0].metadane['waga'] >= 0.6  # Techniczna + sentyment


@pytest.mark.asyncio
async def test_aktualizuj_pozycje(strategia, dane_testowe):
    """Test aktualizacji pozycji."""
    # Mock dla strategii składowych
    mock_tech_sygnal = Mock(
        kierunek=KierunekTransakcji.SHORT,
        metadane={}
    )
    
    with patch.object(strategia.techniczna, 'aktualizuj', return_value=[mock_tech_sygnal]), \
         patch.object(strategia.wyckoff, 'aktualizuj', return_value=[]), \
         patch.object(strategia, '_analizuj_sentyment', return_value={
             'sentyment': -0.5,
             'pewnosc': 0.8,
             'srednia_zmiana': -2.5,
             'liczba_wzmianek': 10
         }):
        
        sygnaly = await strategia.aktualizuj(
            dane_testowe,
            'NKY',
            KierunekTransakcji.LONG,
            100.0
        )
        
        assert len(sygnaly) == 1  # Sygnał zamknięcia
        assert sygnaly[0].kierunek == KierunekTransakcji.SHORT
        assert 'zysk_procent' in sygnaly[0].metadane


def test_optymalizuj(strategia, dane_testowe):
    """Test optymalizacji parametrów."""
    parametry_zakres = {
        'rsi_okres': (10, 20, 2),
        'macd_szybki': (8, 16, 2)
    }
    
    with patch.object(strategia.techniczna, 'optymalizuj', return_value={'rsi_okres': 14}), \
         patch.object(strategia.wyckoff, 'optymalizuj', return_value={'macd_szybki': 12}):
        
        wynik = strategia.optymalizuj(dane_testowe, parametry_zakres)
        
        assert 'rsi_okres' in wynik
        assert 'macd_szybki' in wynik
        assert 'waga_techniczna' in wynik


def test_generuj_statystyki(strategia):
    """Test generowania statystyk."""
    historia = [
        Mock(
            kierunek=KierunekTransakcji.LONG,
            metadane={'zysk_procent': 2.5, 'sentyment': 0.5, 'sentyment_pewnosc': 0.8}
        ),
        Mock(
            kierunek=KierunekTransakcji.SHORT,
            metadane={'zysk_procent': -1.5, 'sentyment': -0.3, 'sentyment_pewnosc': 0.7}
        ),
        Mock(
            kierunek=KierunekTransakcji.LONG,
            metadane={'zysk_procent': 3.0, 'sentyment': 0.6, 'sentyment_pewnosc': 0.9}
        )
    ]
    
    statystyki = strategia.generuj_statystyki(historia)
    
    assert statystyki['liczba_transakcji'] == 3
    assert statystyki['win_rate'] == pytest.approx(0.67, rel=0.01)
    assert statystyki['liczba_long'] == 2
    assert statystyki['liczba_short'] == 1
    assert statystyki['sredni_zysk'] > 0 