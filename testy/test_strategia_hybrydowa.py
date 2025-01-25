import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strategie.hybrydowa import StrategiaHybrydowa
from strategie.interfejs import KierunekTransakcji
from baza_danych.modele import KalendarzEkonomiczny
from backtesting.symulator import SymulatorRynku

@pytest.fixture
def dane_testowe():
    """Przygotowuje testowe dane OHLCV."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1D')
    data = {
        'open': np.random.uniform(100, 110, len(dates)),
        'high': np.random.uniform(110, 120, len(dates)),
        'low': np.random.uniform(90, 100, len(dates)),
        'close': np.random.uniform(100, 110, len(dates)),
        'volume': np.random.uniform(1000, 2000, len(dates))
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def strategia():
    """Tworzy instancję strategii z mockami."""
    mock_baza = Mock()
    mock_baza.session = Mock()
    mock_tech = Mock()
    mock_wyckoff = Mock()
    mock_rag = Mock()
    
    return StrategiaHybrydowa(
        baza=mock_baza,
        techniczna=mock_tech,
        wyckoff=mock_wyckoff,
        rag=mock_rag
    )

@pytest.mark.asyncio
async def test_analizuj_wydarzenia_kalendarz_brak_wydarzen(strategia, dane_testowe):
    """Test analizy kalendarza gdy brak wydarzeń."""
    strategia.baza.session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
    
    wynik = await strategia._analizuj_wydarzenia_kalendarz('USDJPY', dane_testowe)
    
    assert wynik['wplyw'] == 0
    assert wynik['waznosc'] == 0
    assert wynik['liczba'] == 0
    assert wynik['opis'] == "Brak ważnych wydarzeń"

@pytest.mark.asyncio
async def test_analizuj_wydarzenia_kalendarz_z_wydarzeniami(strategia, dane_testowe):
    """Test analizy kalendarza z istniejącymi wydarzeniami."""
    wydarzenia = [
        Mock(
            timestamp=dane_testowe.index[-1],
            nazwa="Test Event 1",
            waluta="JPY",
            waznosc=3,
            wartosc_aktualna=1.5,
            wartosc_prognoza=1.0
        ),
        Mock(
            timestamp=dane_testowe.index[-1],
            nazwa="Test Event 2",
            waluta="JPY",
            waznosc=2,
            wartosc_aktualna=0.8,
            wartosc_prognoza=1.0
        )
    ]
    
    strategia.baza.session.query.return_value.filter.return_value.order_by.return_value.all.return_value = wydarzenia
    
    wynik = await strategia._analizuj_wydarzenia_kalendarz('USDJPY', dane_testowe)
    
    assert wynik['liczba'] == 2
    assert wynik['waznosc'] == 2.5  # (3 + 2) / 2
    assert 'wplyw' in wynik
    assert isinstance(wynik['opis'], str)
    assert "Test Event 1" in wynik['opis']
    assert "Test Event 2" in wynik['opis']

@pytest.mark.asyncio
async def test_analizuj_sygnaly_z_kalendarzem(strategia, dane_testowe):
    """Test generowania sygnałów z uwzględnieniem kalendarza."""
    # Mockuj sygnały techniczne
    async def mock_generuj_sygnaly(*args, **kwargs):
        return [Mock(kierunek=KierunekTransakcji.LONG)]
    strategia.techniczna.generuj_sygnaly = mock_generuj_sygnaly
    
    # Mockuj sygnały Wyckoffa
    async def mock_analizuj_sygnaly(*args, **kwargs):
        return [Mock(kierunek=KierunekTransakcji.LONG)]
    strategia.wyckoff.analizuj_sygnaly = mock_analizuj_sygnaly
    
    # Mockuj wydarzenia z kalendarza
    wydarzenia = [
        Mock(
            timestamp=dane_testowe.index[-1],
            nazwa="Test Event",
            waluta="JPY",
            waznosc=3,
            wartosc_aktualna=1.5,
            wartosc_prognoza=1.0
        )
    ]
    strategia.baza.session.query.return_value.filter.return_value.order_by.return_value.all.return_value = wydarzenia
    
    # Mockuj RAG
    async def mock_znajdz_wzorce(*args, **kwargs):
        return [Mock()]
    strategia.rag.znajdz_podobne_wzorce = mock_znajdz_wzorce
    
    # Mockuj analizę sentymentu
    async def mock_sentyment(*args, **kwargs):
        return {
            'sentyment': 0.8,
            'pewnosc': 0.9,
            'opis': 'Pozytywny sentyment'
        }
    strategia._analizuj_sentyment = mock_sentyment
    
    # Mockuj obliczanie ATR
    strategia.techniczna.oblicz_atr.return_value = np.array([0.5])
    
    # Wykonaj test
    sygnaly = await strategia.analizuj_sygnaly('USDJPY', dane_testowe)
    
    assert len(sygnaly) == 1
    assert sygnaly[0].kierunek == KierunekTransakcji.LONG
    assert "Wydarzenia ekonomiczne" in sygnaly[0].opis
    assert "Sentyment rynku" in sygnaly[0].opis
    assert sygnaly[0].wolumen > 0
    assert sygnaly[0].wolumen <= 1.0
    assert sygnaly[0].stop_loss < sygnaly[0].cena_wejscia
    assert sygnaly[0].take_profit > sygnaly[0].cena_wejscia

@pytest.mark.asyncio
async def test_pobierz_walute_dla_symbolu(strategia):
    """Test pobierania waluty dla różnych symboli."""
    assert strategia._pobierz_walute_dla_symbolu('JP225') == 'JPY'
    assert strategia._pobierz_walute_dla_symbolu('USDJPY') == 'JPY'
    assert strategia._pobierz_walute_dla_symbolu('UNKNOWN') is None

def test_inicjalizuj(strategia):
    """Test inicjalizacji parametrów strategii."""
    # Przygotuj parametry testowe
    parametry = {
        'waga_techniczna': 0.5,
        'waga_wyckoff': 0.3,
        'waga_sentyment': 0.2,
        'min_pewnosc_sentymentu': 0.8,
        'min_podobienstwo_wzorca': 0.9,
        'sl_atr': 2.5,
        'tp_atr': 3.5
    }
    
    # Inicjalizuj strategię
    strategia.inicjalizuj(parametry)
    
    # Sprawdź czy parametry zostały zaktualizowane
    assert strategia.parametry['waga_techniczna'] == 0.5
    assert strategia.parametry['waga_wyckoff'] == 0.3
    assert strategia.parametry['waga_sentyment'] == 0.2
    assert strategia.parametry['min_pewnosc_sentymentu'] == 0.8
    assert strategia.parametry['min_podobienstwo_wzorca'] == 0.9
    assert strategia.parametry['sl_atr'] == 2.5
    assert strategia.parametry['tp_atr'] == 3.5
    
    # Sprawdź czy strategie składowe zostały zainicjalizowane
    strategia.techniczna.inicjalizuj.assert_called_once_with(parametry)
    strategia.wyckoff.inicjalizuj.assert_called_once_with(parametry)

def test_generuj_statystyki_pusta_historia(strategia):
    """Test generowania statystyk dla pustej historii."""
    statystyki = strategia.generuj_statystyki([])
    assert statystyki == {}

def test_generuj_statystyki(strategia):
    """Test generowania statystyk dla historii transakcji."""
    # Przygotuj testową historię
    historia = [
        Mock(
            kierunek=KierunekTransakcji.LONG,
            metadane={
                'zysk_procent': 2.5,
                'sentyment': 0.8,
                'sentyment_pewnosc': 0.9
            }
        ),
        Mock(
            kierunek=KierunekTransakcji.SHORT,
            metadane={
                'zysk_procent': -1.5,
                'sentyment': -0.3,
                'sentyment_pewnosc': 0.7
            }
        ),
        Mock(
            kierunek=KierunekTransakcji.LONG,
            metadane={
                'zysk_procent': 1.8,
                'sentyment': 0.6,
                'sentyment_pewnosc': 0.8
            }
        )
    ]
    
    statystyki = strategia.generuj_statystyki(historia)
    
    assert statystyki['liczba_transakcji'] == 3
    assert statystyki['win_rate'] == pytest.approx(2/3)
    assert statystyki['liczba_long'] == 2
    assert statystyki['liczba_short'] == 1
    assert statystyki['win_rate_long'] == 1.0
    assert statystyki['win_rate_short'] == 0.0
    assert statystyki['sredni_sentyment'] == pytest.approx(0.367, abs=0.001)
    assert statystyki['srednia_pewnosc_sentymentu'] == pytest.approx(0.8, abs=0.001)
    assert statystyki['sredni_zysk'] == pytest.approx(0.933, abs=0.001)
    assert statystyki['max_zysk'] == 2.5
    assert statystyki['max_strata'] == -1.5

@pytest.mark.asyncio
async def test_analizuj(strategia, dane_testowe):
    """Test analizy danych i generowania sygnałów."""
    # Mockuj sygnały techniczne
    mock_sygnal_tech = Mock(
        kierunek=KierunekTransakcji.LONG,
        metadane={'symbol': 'USDJPY'}
    )
    strategia.techniczna.analizuj.return_value = [mock_sygnal_tech]

    # Mockuj sygnały Wyckoffa
    mock_sygnal_wyckoff = Mock(
        kierunek=KierunekTransakcji.LONG,
        metadane={'symbol': 'USDJPY'}
    )
    strategia.wyckoff.analizuj.return_value = [mock_sygnal_wyckoff]

    # Mockuj analizę sentymentu
    async def mock_sentyment(*args, **kwargs):
        return {
            'sentyment': 0.8,
            'pewnosc': 0.9,
            'liczba_wzmianek': 100,
            'opis': 'Pozytywny sentyment'
        }
    strategia._analizuj_sentyment = mock_sentyment

    # Mockuj analizę wydarzeń
    async def mock_wydarzenia(*args, **kwargs):
        return {
            'wplyw': 0.5,
            'waznosc': 2,
            'liczba': 1,
            'opis': 'Ważne wydarzenie'
        }
    strategia._analizuj_wydarzenia_kalendarz = mock_wydarzenia

    # Mockuj obliczanie ATR
    strategia.techniczna.oblicz_atr.return_value = np.array([0.5])

    # Wykonaj test
    sygnaly = await strategia.analizuj(dane_testowe, symbol='USDJPY')

    assert len(sygnaly) == 1
    assert sygnaly[0].kierunek == KierunekTransakcji.LONG
    assert sygnaly[0].metadane['symbol'] == 'USDJPY'
    assert 'sentyment' in sygnaly[0].metadane
    assert 'wydarzenia' in sygnaly[0].metadane

@pytest.mark.asyncio
async def test_aktualizuj(strategia, dane_testowe):
    """Test aktualizacji pozycji."""
    # Mockuj sygnały techniczne
    mock_sygnal_tech = Mock(
        kierunek=KierunekTransakcji.SHORT,
        metadane={'symbol': 'USDJPY'}
    )
    strategia.techniczna.aktualizuj.return_value = [mock_sygnal_tech]

    # Mockuj sygnały Wyckoffa
    mock_sygnal_wyckoff = Mock(
        kierunek=KierunekTransakcji.SHORT,
        metadane={'symbol': 'USDJPY'}
    )
    strategia.wyckoff.aktualizuj.return_value = [mock_sygnal_wyckoff]

    # Mockuj analizę sentymentu
    async def mock_sentyment(*args, **kwargs):
        return {
            'sentyment': -0.8,
            'pewnosc': 0.9,
            'liczba_wzmianek': 50,
            'opis': 'Negatywny sentyment'
        }
    strategia._analizuj_sentyment = mock_sentyment

    # Wykonaj test dla aktywnej pozycji LONG
    sygnaly = await strategia.aktualizuj(
        dane_testowe,
        symbol='USDJPY',
        kierunek=KierunekTransakcji.LONG,
        cena_wejscia=100.0
    )

    assert len(sygnaly) == 1
    assert sygnaly[0].kierunek == KierunekTransakcji.SHORT
    assert sygnaly[0].metadane['symbol'] == 'USDJPY'
    assert 'sentyment' in sygnaly[0].metadane
    assert sygnaly[0].metadane['sentyment'] < 0

def test_optymalizuj(strategia, dane_testowe):
    """Test optymalizacji parametrów strategii."""
    # Mockuj wyniki optymalizacji strategii składowych
    strategia.techniczna.optymalizuj.return_value = {
        'rsi_okres': 14,
        'rsi_wykupienie': 70,
        'rsi_wyprzedanie': 30
    }

    strategia.wyckoff.optymalizuj.return_value = {
        'vol_std_mult': 2.0,
        'trend_momentum': 0.5
    }

    # Mockuj wyniki backtestów
    class MockWynikBacktestu:
        def __init__(self, zysk, win_rate, profit_factor, max_drawdown, sharpe_ratio):
            self.zysk_procent = zysk
            self.win_rate = win_rate
            self.profit_factor = profit_factor
            self.max_drawdown = max_drawdown
            self.sharpe_ratio = sharpe_ratio

    mock_wynik = MockWynikBacktestu(
        zysk=15.0,
        win_rate=0.6,
        profit_factor=1.5,
        max_drawdown=10.0,
        sharpe_ratio=1.2
    )

    with patch('backtesting.symulator.SymulatorRynku') as mock_symulator:
        mock_symulator.return_value.testuj_strategie.return_value = mock_wynik
        
        # Wykonaj test
        parametry = strategia.optymalizuj(dane_testowe)

        # Sprawdź czy parametry zostały zoptymalizowane
        assert isinstance(parametry, dict)
        assert 'waga_techniczna' in parametry
        assert 'waga_wyckoff' in parametry
        assert 'waga_sentyment' in parametry
        assert 'min_sila_sygnalu' in parametry
        
        # Sprawdź czy strategie składowe zostały zoptymalizowane
        strategia.techniczna.optymalizuj.assert_called_once()
        strategia.wyckoff.optymalizuj.assert_called_once() 