"""
Testy dla strategii Wyckoffa.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock

from strategie.wyckoff import StrategiaWyckoff, FazaWyckoff
from strategie.interfejs import KierunekTransakcji


@pytest.fixture
def dane_testowe():
    """Generuje dane testowe dla różnych faz rynku."""
    daty = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Faza akumulacji (trend boczny, niskie ceny, rosnący wolumen)
    akumulacja = pd.DataFrame({
        'open': np.linspace(98, 99, 20) + np.random.uniform(-0.2, 0.2, 20),  # Mniejsza zmienność
        'high': np.linspace(99, 100, 20) + np.random.uniform(-0.2, 0.2, 20),
        'low': np.linspace(97, 98, 20) + np.random.uniform(-0.2, 0.2, 20),
        'close': np.linspace(98, 99, 20) + np.random.uniform(-0.2, 0.2, 20),
        'volume': np.linspace(1000, 3000, 20),  # Wyraźnie rosnący wolumen
        'symbol': 'NKY'
    }, index=daty[:20])
    
    # Faza wzrostu (rosnące ceny i wolumen)
    wzrost = pd.DataFrame({
        'open': np.linspace(101, 120, 20),  # Wyraźny trend wzrostowy
        'high': np.linspace(102, 122, 20),
        'low': np.linspace(100, 119, 20),
        'close': np.linspace(101, 121, 20),
        'volume': np.linspace(2000, 4000, 20),  # Rosnący wolumen
        'symbol': 'NKY'
    }, index=daty[20:40])
    
    # Faza dystrybucji (trend boczny, wysokie ceny, spadający wolumen)
    dystrybucja = pd.DataFrame({
        'open': np.linspace(120, 121, 20) + np.random.uniform(-0.2, 0.2, 20),  # Mniejsza zmienność
        'high': np.linspace(121, 122, 20) + np.random.uniform(-0.2, 0.2, 20),
        'low': np.linspace(119, 120, 20) + np.random.uniform(-0.2, 0.2, 20),
        'close': np.linspace(120, 121, 20) + np.random.uniform(-0.2, 0.2, 20),
        'volume': np.linspace(4000, 2000, 20),  # Wyraźnie spadający wolumen
        'symbol': 'NKY'
    }, index=daty[40:60])
    
    # Faza spadku (spadające ceny i wolumen)
    spadek = pd.DataFrame({
        'open': np.linspace(121, 102, 20),  # Wyraźny trend spadkowy
        'high': np.linspace(122, 103, 20),
        'low': np.linspace(120, 101, 20),
        'close': np.linspace(121, 102, 20),
        'volume': np.linspace(2000, 1000, 20),  # Spadający wolumen
        'symbol': 'NKY'
    }, index=daty[60:80])
    
    return pd.concat([akumulacja, wzrost, dystrybucja, spadek])


@pytest.fixture
def strategia():
    """Tworzy instancję strategii z domyślnymi parametrami."""
    strategia = StrategiaWyckoff()
    strategia.inicjalizuj({})
    return strategia


def test_identyfikacja_fazy_akumulacji(strategia, dane_testowe):
    """Test identyfikacji fazy akumulacji."""
    df = dane_testowe[:20].copy()  # Pierwsze 20 świec
    faza = strategia._identyfikuj_faze(df)
    assert faza == FazaWyckoff.AKUMULACJA


def test_identyfikacja_fazy_wzrostu(strategia, dane_testowe):
    """Test identyfikacji fazy wzrostu."""
    df = dane_testowe[20:40].copy()  # Świece 20-40
    faza = strategia._identyfikuj_faze(df)
    assert faza == FazaWyckoff.WZROST


def test_identyfikacja_fazy_dystrybucji(strategia, dane_testowe):
    """Test identyfikacji fazy dystrybucji."""
    df = dane_testowe[40:60].copy()  # Świece 40-60
    faza = strategia._identyfikuj_faze(df)
    assert faza == FazaWyckoff.DYSTRYBUCJA


def test_identyfikacja_fazy_spadku(strategia, dane_testowe):
    """Test identyfikacji fazy spadku."""
    df = dane_testowe[60:80].copy()  # Świece 60-80
    faza = strategia._identyfikuj_faze(df)
    assert faza == FazaWyckoff.SPADEK


def test_wykryj_spring(strategia, dane_testowe):
    """Test wykrywania formacji spring."""
    df = dane_testowe[:20].copy()  # Faza akumulacji
    
    # Ustawiamy trend spadkowy przed formacją
    df.loc[df.index[-20:-5], 'close'] = np.linspace(105, 98, 15)  # Trend spadkowy
    df.loc[df.index[-20:-5], 'high'] = df.loc[df.index[-20:-5], 'close'] + 1
    df.loc[df.index[-20:-5], 'low'] = df.loc[df.index[-20:-5], 'close'] - 1
    df.loc[df.index[-20:-5], 'volume'] = np.linspace(1000, 1500, 15)  # Normalny wolumen
    
    # Modyfikujemy ostatnie 5 świec aby utworzyć formację spring
    ostatnie_5_indeksy = df.index[-5:]
    df.loc[ostatnie_5_indeksy, 'low'] = [97, 96, 95, 94, 93]  # Spadające dołki
    df.loc[ostatnie_5_indeksy, 'high'] = [99, 98, 97, 98, 100]  # Wysokie odbicie
    df.loc[ostatnie_5_indeksy, 'close'] = [98, 97, 96, 97, 99]  # Zamknięcia
    df.loc[ostatnie_5_indeksy, 'open'] = [98.5, 97.5, 96.5, 96, 97]  # Otwarcia
    df.loc[ostatnie_5_indeksy, 'volume'] = [1500, 1600, 1700, 1800, 3000]  # Rosnący wolumen z wybiciem
    
    # Dodajemy RSI poniżej 30 z tendencją wzrostową
    rsi = np.linspace(25, 35, len(df))  # RSI rosnący z 25 do 35
    df['rsi'] = rsi
    
    assert strategia._wykryj_spring(df, len(df)-1)


def test_wykryj_upthrust(strategia, dane_testowe):
    """Test wykrywania formacji upthrust."""
    df = dane_testowe[40:60].copy()  # Faza dystrybucji
    
    # Ustawiamy trend wzrostowy przed formacją
    df.loc[df.index[-20:-5], 'close'] = np.linspace(115, 122, 15)  # Trend wzrostowy
    df.loc[df.index[-20:-5], 'high'] = df.loc[df.index[-20:-5], 'close'] + 1
    df.loc[df.index[-20:-5], 'low'] = df.loc[df.index[-20:-5], 'close'] - 1
    df.loc[df.index[-20:-5], 'volume'] = np.linspace(4000, 3000, 15)  # Spadający wolumen
    
    # Modyfikujemy ostatnie 5 świec aby utworzyć formację upthrust
    ostatnie_5_indeksy = df.index[-5:]
    df.loc[ostatnie_5_indeksy, 'high'] = [123, 124, 125, 126, 127]  # Rosnące szczyty
    df.loc[ostatnie_5_indeksy, 'low'] = [121, 122, 123, 122, 120]  # Spadające dołki
    df.loc[ostatnie_5_indeksy, 'close'] = [122, 123, 124, 123, 121]  # Zamknięcia
    df.loc[ostatnie_5_indeksy, 'open'] = [121.5, 122.5, 123.5, 124, 123]  # Otwarcia
    df.loc[ostatnie_5_indeksy, 'volume'] = [2800, 2600, 2400, 2200, 4000]  # Spadający wolumen z wybiciem
    
    # Dodajemy RSI powyżej 70 z tendencją spadkową
    rsi = np.linspace(80, 70, len(df))  # RSI spadający z 80 do 70
    df['rsi'] = rsi
    
    assert strategia._wykryj_upthrust(df, len(df)-1)


def test_generowanie_sygnalu_long(strategia, dane_testowe):
    """Test generowania sygnału LONG w fazie akumulacji."""
    df = dane_testowe[:20].copy()  # Faza akumulacji
    
    # Ustawiamy warunki dla fazy akumulacji
    df['close'] = 98.0  # Niskie ceny
    df['volume'] = np.linspace(1000, 2000, len(df))  # Rosnący wolumen
    
    # Symulujemy trend boczny przed formacją
    df.loc[df.index[-20:-5], 'close'] = np.linspace(98, 98, 15)  # Trend boczny
    df.loc[df.index[-20:-5], 'high'] = df.loc[df.index[-20:-5], 'close'] + 0.5
    df.loc[df.index[-20:-5], 'low'] = df.loc[df.index[-20:-5], 'close'] - 0.5
    df.loc[df.index[-20:-5], 'volume'] = np.linspace(1000, 1500, 15)  # Rosnący wolumen
    
    # Tworzymy formację spring
    ostatnie_5_indeksy = df.index[-5:]
    df.loc[ostatnie_5_indeksy, 'low'] = [97, 96, 95, 94, 93]  # Spadające dołki
    df.loc[ostatnie_5_indeksy, 'high'] = [99, 98, 97, 98, 100]  # Wysokie odbicie
    df.loc[ostatnie_5_indeksy, 'close'] = [98, 97, 96, 97, 99]  # Zamknięcia
    df.loc[ostatnie_5_indeksy, 'open'] = [98.5, 97.5, 96.5, 96, 97]  # Otwarcia
    df.loc[ostatnie_5_indeksy, 'volume'] = [1500, 1600, 1700, 1800, 3000]  # Rosnący wolumen z wybiciem
    
    # Dodajemy RSI poniżej 30 z tendencją wzrostową
    rsi = np.linspace(25, 35, len(df))  # RSI rosnący z 25 do 35
    df['rsi'] = rsi
    
    sygnaly = strategia.analizuj(df)
    
    assert len(sygnaly) == 1
    assert sygnaly[0].kierunek == KierunekTransakcji.LONG
    assert sygnaly[0].metadane['formacja'] == 'spring'
    assert sygnaly[0].metadane['faza'] == FazaWyckoff.AKUMULACJA.value


def test_generowanie_sygnalu_short(strategia, dane_testowe):
    """Test generowania sygnału SHORT w fazie dystrybucji."""
    df = dane_testowe[40:60].copy()  # Faza dystrybucji
    
    # Ustawiamy warunki dla fazy dystrybucji
    df['close'] = 120.0  # Wysokie ceny
    df['volume'] = np.linspace(4000, 2000, len(df))  # Spadający wolumen
    
    # Symulujemy trend boczny przed dystrybucją
    df.loc[df.index[-20:-5], 'close'] = np.linspace(120, 120, 15)  # Trend boczny
    df.loc[df.index[-20:-5], 'high'] = df.loc[df.index[-20:-5], 'close'] + 0.5
    df.loc[df.index[-20:-5], 'low'] = df.loc[df.index[-20:-5], 'close'] - 0.5
    df.loc[df.index[-20:-5], 'volume'] = np.linspace(4000, 3000, 15)  # Spadający wolumen
    
    # Tworzymy formację upthrust
    ostatnie_5_indeksy = df.index[-5:]
    df.loc[ostatnie_5_indeksy, 'high'] = [123, 124, 125, 126, 127]  # Rosnące szczyty
    df.loc[ostatnie_5_indeksy, 'low'] = [121, 122, 123, 122, 120]  # Spadające dołki
    df.loc[ostatnie_5_indeksy, 'close'] = [122, 123, 124, 123, 121]  # Zamknięcia
    df.loc[ostatnie_5_indeksy, 'open'] = [121.5, 122.5, 123.5, 124, 123]  # Otwarcia
    df.loc[ostatnie_5_indeksy, 'volume'] = [2800, 2600, 2400, 2200, 4000]  # Spadający wolumen z wybiciem
    
    # Dodajemy RSI powyżej 70 z tendencją spadkową
    rsi = np.linspace(80, 70, len(df))  # RSI spadający z 80 do 70
    df['rsi'] = rsi
    
    sygnaly = strategia.analizuj(df)
    
    assert len(sygnaly) == 1
    assert sygnaly[0].kierunek == KierunekTransakcji.SHORT
    assert sygnaly[0].metadane['formacja'] == 'upthrust'
    assert sygnaly[0].metadane['faza'] == FazaWyckoff.DYSTRYBUCJA.value


def test_zamkniecie_long_w_dystrybucji(strategia, dane_testowe):
    """Test zamykania pozycji LONG w fazie dystrybucji."""
    df = dane_testowe[40:60].copy()  # Faza dystrybucji
    aktywne_pozycje = [('NKY', KierunekTransakcji.LONG, 100.0)]
    
    sygnaly = strategia.aktualizuj(df, aktywne_pozycje)
    
    assert len(sygnaly) == 1
    assert sygnaly[0].kierunek == KierunekTransakcji.SHORT  # Zamknięcie LONG
    assert 'zysk_procent' in sygnaly[0].metadane
    assert sygnaly[0].metadane['faza'] == FazaWyckoff.DYSTRYBUCJA.value


def test_zamkniecie_short_w_akumulacji(strategia, dane_testowe):
    """Test zamykania pozycji SHORT w fazie akumulacji."""
    df = dane_testowe[:20].copy()  # Faza akumulacji
    aktywne_pozycje = [('NKY', KierunekTransakcji.SHORT, 110.0)]
    
    sygnaly = strategia.aktualizuj(df, aktywne_pozycje)
    
    assert len(sygnaly) == 1
    assert sygnaly[0].kierunek == KierunekTransakcji.LONG  # Zamknięcie SHORT
    assert 'zysk_procent' in sygnaly[0].metadane
    assert sygnaly[0].metadane['faza'] == FazaWyckoff.AKUMULACJA.value


def test_generowanie_statystyk(strategia):
    """Test generowania statystyk dla różnych faz rynku."""
    historia = [
        Mock(metadane={
            'zysk_procent': 1.5,
            'faza': FazaWyckoff.AKUMULACJA.value
        }),
        Mock(metadane={
            'zysk_procent': 2.0,
            'faza': FazaWyckoff.WZROST.value
        }),
        Mock(metadane={
            'zysk_procent': -1.0,
            'faza': FazaWyckoff.DYSTRYBUCJA.value
        }),
        Mock(metadane={
            'zysk_procent': -0.5,
            'faza': FazaWyckoff.SPADEK.value
        })
    ]
    
    statystyki = strategia.generuj_statystyki(historia)
    
    assert statystyki['liczba_transakcji'] == 4
    assert statystyki['win_rate'] == 0.5  # 2 zyskowne z 4
    assert statystyki['liczba_transakcji_AKUMULACJA'] == 1
    assert statystyki['zysk_sredni_AKUMULACJA'] == 1.5
    assert statystyki['win_rate_AKUMULACJA'] == 1.0


def test_inicjalizacja_parametrow(strategia):
    """Test inicjalizacji z różnymi parametrami."""
    parametry = {
        'okres_ma': 10,
        'min_spread_mult': 0.3,
        'min_vol_mult': 1.5,
        'sl_atr': 3.0,
        'tp_atr': 4.0,
        'vol_std_mult': 2.0,
        'min_swing_candles': 5,
        'rsi_okres': 21,
        'rsi_min': 25,
        'rsi_max': 75,
        'trend_momentum': 0.2
    }
    
    strategia.inicjalizuj(parametry)
    
    for nazwa, wartosc in parametry.items():
        assert strategia.parametry[nazwa] == wartosc


def test_obsluga_blednych_danych(strategia):
    """Test zachowania przy nieprawidłowych danych."""
    # Test pustego DataFrame
    df_pusty = pd.DataFrame()
    sygnaly = strategia.analizuj(df_pusty)
    assert len(sygnaly) == 0
    
    # Test niepełnych danych
    df_niepelny = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200]
    })
    sygnaly = strategia.analizuj(df_niepelny)
    assert len(sygnaly) == 0
    
    # Test nieprawidłowych wartości
    df_bledny = pd.DataFrame({
        'open': [np.nan, 100, 101],
        'high': [102, np.inf, 103],
        'low': [98, -np.inf, 99],
        'close': [101, None, 102],
        'volume': [-1000, 0, 1100]
    })
    sygnaly = strategia.analizuj(df_bledny)
    assert len(sygnaly) == 0


def test_przypadki_graniczne_rsi(strategia, dane_testowe):
    """Test zachowania przy granicznych wartościach RSI."""
    df = dane_testowe[:20].copy()
    
    # Test przy RSI = 0
    df['rsi'] = 0
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) == 0
    
    # Test przy RSI = 100
    df['rsi'] = 100
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) == 0
    
    # Test przy RSI = 50 (neutralny)
    df['rsi'] = 50
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) == 0


def test_przypadki_graniczne_wolumenu(strategia, dane_testowe):
    """Test zachowania przy ekstremalnych wolumenach."""
    df = dane_testowe[:20].copy()
    
    # Test przy zerowym wolumenie
    df['volume'] = 0
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) == 0
    
    # Test przy bardzo wysokim wolumenie
    df['volume'] = 1e9
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) == 0


def test_optymalizacja_parametrow(strategia, dane_testowe):
    """Test optymalizacji parametrów."""
    parametry_zakres = {
        'okres_ma': [10, 20, 30],
        'min_spread_mult': [0.3, 0.5, 0.7],
        'min_vol_mult': [1.1, 1.3, 1.5],
        'trend_momentum': [0.05, 0.1, 0.15]
    }
    
    wynik = strategia.optymalizuj(dane_testowe, parametry_zakres)
    
    assert isinstance(wynik, dict)
    for param in parametry_zakres.keys():
        assert param in wynik
        assert wynik[param] in parametry_zakres[param]


def test_nietypowy_spring(strategia, dane_testowe):
    """Test wykrywania nietypowej formacji spring z dużą zmiennością."""
    df = dane_testowe[:20].copy()
    
    # Symulujemy bardzo zmienną cenę przed formacją
    df.loc[df.index[-20:-5], 'close'] = np.array([100, 98, 102, 97, 103, 96, 101, 95, 100, 94, 99, 93, 98, 92, 97])
    df.loc[df.index[-20:-5], 'high'] = df.loc[df.index[-20:-5], 'close'] + 2
    df.loc[df.index[-20:-5], 'low'] = df.loc[df.index[-20:-5], 'close'] - 2
    df.loc[df.index[-20:-5], 'volume'] = np.random.uniform(1000, 2000, 15)
    
    # Tworzymy nietypowy spring z bardzo dużym wolumenem
    ostatnie_5_indeksy = df.index[-5:]
    df.loc[ostatnie_5_indeksy, 'low'] = [91, 90, 88, 85, 84]  # Głęboki spring
    df.loc[ostatnie_5_indeksy, 'high'] = [95, 94, 92, 93, 98]  # Silne odbicie
    df.loc[ostatnie_5_indeksy, 'close'] = [93, 92, 90, 92, 97]  # Mocne zamknięcie
    df.loc[ostatnie_5_indeksy, 'open'] = [94, 93, 91, 85, 86]  # Duża zmienność
    df.loc[ostatnie_5_indeksy, 'volume'] = [2000, 2500, 3000, 5000, 10000]  # Ekstremalny wolumen
    
    df['rsi'] = np.concatenate([np.linspace(40, 20, 15), np.linspace(20, 35, 5)])
    
    assert strategia._wykryj_spring(df, len(df)-1)


def test_nietypowy_upthrust(strategia, dane_testowe):
    """Test wykrywania nietypowej formacji upthrust z dużą zmiennością."""
    df = dane_testowe[40:60].copy()
    
    # Symulujemy bardzo zmienną cenę przed formacją
    df.loc[df.index[-20:-5], 'close'] = np.array([120, 122, 118, 123, 117, 124, 116, 125, 115, 126, 114, 127, 113, 128, 112])
    df.loc[df.index[-20:-5], 'high'] = df.loc[df.index[-20:-5], 'close'] + 2
    df.loc[df.index[-20:-5], 'low'] = df.loc[df.index[-20:-5], 'close'] - 2
    df.loc[df.index[-20:-5], 'volume'] = np.random.uniform(3000, 4000, 15)
    
    # Tworzymy nietypowy upthrust z bardzo dużym wolumenem
    ostatnie_5_indeksy = df.index[-5:]
    df.loc[ostatnie_5_indeksy, 'high'] = [130, 132, 135, 140, 145]  # Ekstremalne wybicie
    df.loc[ostatnie_5_indeksy, 'low'] = [125, 127, 130, 132, 125]  # Głęboki spadek
    df.loc[ostatnie_5_indeksy, 'close'] = [127, 129, 132, 135, 126]  # Silne zamknięcie w dół
    df.loc[ostatnie_5_indeksy, 'open'] = [126, 128, 131, 133, 140]  # Duża zmienność
    df.loc[ostatnie_5_indeksy, 'volume'] = [4000, 4500, 5000, 7000, 15000]  # Ekstremalny wolumen
    
    df['rsi'] = np.concatenate([np.linspace(60, 80, 15), np.linspace(80, 70, 5)])
    
    assert strategia._wykryj_upthrust(df, len(df)-1)


def test_zmiana_fazy_w_trakcie_formacji(strategia, dane_testowe):
    """Test zachowania gdy faza rynku zmienia się w trakcie formacji."""
    df = dane_testowe[:30].copy()
    
    # Pierwsza połowa - faza akumulacji
    df.loc[df.index[:15], 'close'] = 98.0
    df.loc[df.index[:15], 'volume'] = np.linspace(1000, 2000, 15)
    df.loc[df.index[:15], 'rsi'] = np.linspace(30, 35, 15)
    
    # Druga połowa - przejście do fazy wzrostu
    df.loc[df.index[15:], 'close'] = np.linspace(98, 105, 15)
    df.loc[df.index[15:], 'volume'] = np.linspace(2000, 3000, 15)
    df.loc[df.index[15:], 'rsi'] = np.linspace(35, 60, 15)
    
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) == 0  # Nie powinno być sygnału przy niestabilnej fazie


def test_wielokrotne_formacje(strategia, dane_testowe):
    """Test wykrywania wielokrotnych formacji w krótkim czasie."""
    df = dane_testowe[:30].copy()
    
    # Tworzymy dwa springi blisko siebie
    df.loc[df.index[10:15], 'low'] = [94, 93, 92, 94, 96]  # Pierwszy spring
    df.loc[df.index[10:15], 'close'] = [95, 94, 93, 95, 97]
    df.loc[df.index[10:15], 'volume'] = [2000, 2500, 3000, 3500, 4000]
    
    df.loc[df.index[-5:], 'low'] = [93, 92, 91, 93, 95]  # Drugi spring
    df.loc[df.index[-5:], 'close'] = [94, 93, 92, 94, 96]
    df.loc[df.index[-5:], 'volume'] = [2500, 3000, 3500, 4000, 4500]
    
    df['rsi'] = 30  # Stały niski RSI
    
    # Powinien wykryć tylko jeden sygnał (unikanie nadmiernego tradingu)
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) <= 1


def test_zachowanie_trendu_po_formacji(strategia, dane_testowe):
    """Test sprawdzający zachowanie po formacji."""
    df = dane_testowe[:30].copy()

    # Tworzymy trend boczny przed springiem
    df.loc[df.index[:15], 'close'] = np.linspace(98, 98, 15)  # Trend boczny
    df.loc[df.index[:15], 'volume'] = np.linspace(2000, 2500, 15)  # Rosnący wolumen
    df.loc[df.index[:15], 'rsi'] = np.linspace(30, 32, 15)  # Niski RSI
    df.loc[df.index[:15], 'open'] = df.loc[df.index[:15], 'close']
    df.loc[df.index[:15], 'high'] = df.loc[df.index[:15], 'close'] + 1
    df.loc[df.index[:15], 'low'] = df.loc[df.index[:15], 'close'] - 1

    # Tworzymy spring
    df.loc[df.index[15:20], 'low'] = [97, 96, 95, 94, 93]  # Nowe minimum
    df.loc[df.index[15:20], 'close'] = [98, 97, 96, 95, 99]  # Odbicie
    df.loc[df.index[15:20], 'open'] = [99, 98, 97, 94, 97]  # Świece wzrostowe
    df.loc[df.index[15:20], 'high'] = [100, 99, 98, 96, 100]
    df.loc[df.index[15:20], 'volume'] = [2500, 3000, 3500, 4000, 4500]  # Rosnący wolumen
    df.loc[df.index[15:20], 'rsi'] = [32, 31, 30, 32, 35]  # Niski RSI z odbiciem

    # Sprawdzamy sygnał po utworzeniu springa
    sygnaly = strategia.analizuj(df.iloc[:20])
    assert len(sygnaly) == 1
    assert sygnaly[0].kierunek == KierunekTransakcji.LONG
    assert sygnaly[0].metadane['formacja'] == 'spring'

    # Dodajemy dane po formacji - trend wzrostowy
    df.loc[df.index[20:], 'close'] = np.linspace(99, 99.5, 10)  # Bardzo łagodny trend wzrostowy
    df.loc[df.index[20:], 'volume'] = np.linspace(4000, 4200, 10)  # Stabilny wolumen
    df.loc[df.index[20:], 'rsi'] = np.linspace(35, 40, 10)  # Umiarkowany wzrost RSI
    df.loc[df.index[20:], 'open'] = df.loc[df.index[20:], 'close'] - 0.2
    df.loc[df.index[20:], 'high'] = df.loc[df.index[20:], 'close'] + 0.2
    df.loc[df.index[20:], 'low'] = df.loc[df.index[20:], 'close'] - 0.2

    # Sprawdzamy czy sygnał jest zachowany
    sygnaly = strategia.analizuj(df)
    assert len(sygnaly) == 1
    assert sygnaly[0].kierunek == KierunekTransakcji.LONG
    assert sygnaly[0].metadane['formacja'] == 'spring' 