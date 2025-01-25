"""
Testy jednostkowe dla modułu rag_rynkowy.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from ai.rag_rynkowy import RAGRynkowy
from ai.system_rag import SystemRAG
from handel.analiza_techniczna import AnalizaTechniczna
from strategie.wyckoff import StrategiaWyckoff, FazaWyckoff

def generuj_dane_testowe(n_dni: int = 100) -> pd.DataFrame:
    """Generuje testowe dane OHLCV."""
    np.random.seed(42)
    
    # Generuj daty
    daty = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_dni)]
    
    # Generuj ceny
    close = np.random.normal(100, 5, n_dni).cumsum()
    high = close + np.random.uniform(0, 2, n_dni)
    low = close - np.random.uniform(0, 2, n_dni)
    open_price = close - np.random.uniform(-1, 1, n_dni)
    volume = np.random.uniform(1000, 5000, n_dni)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=daty)

@pytest.fixture
def rag_mock():
    """Fixture dostarczająca zamockowany system RAG."""
    mock = Mock(spec=SystemRAG)
    mock.dodaj_dokument.return_value = True
    mock.wyszukaj.return_value = [
        {
            'tekst': 'Przykładowy wzorzec',
            'metadata': {
                'id': 'wzorzec_test',
                'data': '2024-01-01T00:00:00',
                'zmiana_proc': 1.5
            },
            'score': 0.8
        }
    ]
    return mock

@pytest.fixture
def rag_rynkowy(rag_mock):
    """Fixture dostarczająca obiekt RAGRynkowy z zamockowanym systemem RAG."""
    return RAGRynkowy(rag_mock)

def test_przygotuj_opis_wzorca(rag_rynkowy):
    """Test generowania opisu wzorca cenowego."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    
    # Wywołaj metodę
    opis = rag_rynkowy._przygotuj_opis_wzorca(dane, 20, okno=10)
    
    # Sprawdź wynik
    assert isinstance(opis, str)
    assert "Wzorzec cenowy z okresu" in opis
    assert "Zmiana ceny:" in opis
    assert "Średni wolumen:" in opis
    assert "Zmienność (STD H-L):" in opis
    assert "Ceny: Open" in opis
    assert "Wolumen:" in opis

def test_indeksuj_dane_historyczne(rag_rynkowy, rag_mock):
    """Test indeksowania danych historycznych."""
    # Przygotuj dane
    dane = generuj_dane_testowe(50)
    
    # Wywołaj metodę
    wynik = rag_rynkowy.indeksuj_dane_historyczne(dane, okno=10, krok=5)
    
    # Sprawdź wynik
    assert wynik is True
    assert rag_mock.dodaj_dokument.call_count == len(range(10, 50, 5))
    
    # Sprawdź argumenty pierwszego wywołania
    args = rag_mock.dodaj_dokument.call_args_list[0][0]
    assert isinstance(args[0], str)  # opis
    assert isinstance(args[1], dict)  # metadata
    assert 'id' in args[1]
    assert 'data' in args[1]
    assert 'zmiana_proc' in args[1]
    assert 'wolumen_sredni' in args[1]
    assert 'zmiennosc' in args[1]

def test_aktualizuj_dane(rag_rynkowy, rag_mock):
    """Test aktualizacji danych o najnowszy wzorzec."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    
    # Wywołaj metodę
    wynik = rag_rynkowy.aktualizuj_dane(dane, okno=10)
    
    # Sprawdź wynik
    assert wynik is True
    rag_mock.dodaj_dokument.assert_called_once()
    
    # Sprawdź argumenty
    args = rag_mock.dodaj_dokument.call_args[0]
    assert isinstance(args[0], str)  # opis
    assert isinstance(args[1], dict)  # metadata
    assert args[2] is True  # aktualizuj

def test_znajdz_podobne_wzorce(rag_rynkowy, rag_mock):
    """Test wyszukiwania podobnych wzorców."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    
    # Ustaw mock dla wyszukiwania
    rag_mock.wyszukaj.return_value = [
        {
            'tekst': 'Wzorzec testowy',
            'metadata': {'data': '2024-01-01T00:00:00', 'zmiana_proc': 0.5},
            'score': 0.8
        }
    ]

    # Wywołaj metodę
    wyniki = rag_rynkowy.znajdz_podobne_wzorce(
        dane,
        okno=10,
        n_wynikow=5,
        min_zmiana=-1.0,
        max_zmiana=1.0
    )
    
    # Sprawdź wynik
    assert isinstance(wyniki, list)
    assert len(wyniki) == 1  # zgodnie z mockiem
    assert 'tekst' in wyniki[0]
    assert 'metadata' in wyniki[0]
    assert 'score' in wyniki[0]
    
    # Sprawdź argumenty wywołania
    args = rag_mock.wyszukaj.call_args
    assert isinstance(args[0][0], str)  # opis
    assert args[0][1] == 5  # n_wynikow (pobieramy dokładną liczbę wyników)
    assert isinstance(args[0][2], dict)  # filtry

def test_obsluga_bledow(rag_rynkowy, rag_mock):
    """Test obsługi błędów."""
    # Przygotuj dane
    dane = pd.DataFrame()  # Puste dane
    
    # Sprawdź obsługę błędów dla każdej metody
    assert rag_rynkowy.indeksuj_dane_historyczne(dane) is False
    assert rag_rynkowy.aktualizuj_dane(dane) is False
    assert rag_rynkowy.znajdz_podobne_wzorce(dane) == []

def test_walidacja_okna(rag_rynkowy):
    """Test walidacji rozmiaru okna."""
    # Przygotuj dane
    dane = generuj_dane_testowe(5)  # Za mało danych
    okno = 10
    
    # Sprawdź czy metody poprawnie obsługują za małe dane
    assert rag_rynkowy.aktualizuj_dane(dane, okno=okno) is False
    assert rag_rynkowy.znajdz_podobne_wzorce(dane, okno=okno) == [] 

def test_oblicz_wskazniki(rag_rynkowy):
    """Test obliczania wskaźników technicznych."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    
    # Wywołaj metodę
    wskazniki = rag_rynkowy._oblicz_wskazniki(dane, 10, 20)
    
    # Sprawdź wynik
    assert isinstance(wskazniki, dict)
    assert 'rsi' in wskazniki
    assert 'atr' in wskazniki
    assert 'momentum' in wskazniki
    assert 'volatility' in wskazniki
    assert 'rel_volume' in wskazniki
    assert all(isinstance(v, float) for v in wskazniki.values())

def test_wykryj_formacje(rag_rynkowy):
    """Test wykrywania formacji świecowych."""
    # Przygotuj dane testowe dla różnych formacji
    dane = pd.DataFrame({
        'open':  [100, 102, 101, 103, 105],
        'high':  [103, 104, 102, 106, 108],
        'low':   [98,  100, 99,  102, 103],
        'close': [102, 101, 102, 105, 104],
        'volume': [1000] * 5
    }, index=pd.date_range('2024-01-01', periods=5))
    
    # Test Doji
    doji_data = dane.copy()
    doji_data.loc[doji_data.index[-1], 'close'] = doji_data.loc[doji_data.index[-1], 'open']
    formacje = rag_rynkowy._wykryj_formacje(doji_data, 0, len(doji_data)-1)
    assert "Doji" in formacje
    
    # Test Młota
    hammer_data = dane.copy()
    hammer_data.loc[hammer_data.index[-1], ['open', 'close', 'high', 'low']] = [105, 106, 106, 100]
    formacje = rag_rynkowy._wykryj_formacje(hammer_data, 0, len(hammer_data)-1)
    assert "Młot" in formacje
    
    # Test Gwiazdy porannej
    morning_star_data = pd.DataFrame({
        'open':  [105, 102, 101],
        'high':  [106, 103, 105],
        'low':   [102, 101, 100],
        'close': [102, 102, 104],
        'volume': [1000] * 3
    }, index=pd.date_range('2024-01-01', periods=3))
    formacje = rag_rynkowy._wykryj_formacje(morning_star_data, 0, len(morning_star_data)-1)
    assert "Gwiazda poranna" in formacje
    
    # Test Pochłonięcia hossy
    engulfing_data = pd.DataFrame({
        'open':  [105, 102],
        'high':  [106, 106],
        'low':   [102, 101],
        'close': [102, 105],
        'volume': [1000] * 2
    }, index=pd.date_range('2024-01-01', periods=2))
    formacje = rag_rynkowy._wykryj_formacje(engulfing_data, 0, len(engulfing_data)-1)
    assert "Pochłonięcie hossy" in formacje

def test_identyfikuj_faze_wyckoff(rag_rynkowy):
    """Test identyfikacji fazy Wyckoffa."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    
    # Wywołaj metodę
    faza, opis = rag_rynkowy._identyfikuj_faze_wyckoff(dane, 10, 20)
    
    # Sprawdź wynik
    assert isinstance(faza, FazaWyckoff)
    assert isinstance(opis, str)
    assert len(opis) > 0

def test_cache_key(rag_rynkowy):
    """Test generowania klucza cache'a."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    okno = 20
    
    # Wywołaj metodę
    klucz = rag_rynkowy._cache_key(dane, okno)
    
    # Sprawdź wynik
    assert isinstance(klucz, str)
    assert "wzorzec_" in klucz
    assert str(okno) in klucz

def test_cache_wynik(rag_rynkowy):
    """Test zapisywania wyników w cache'u."""
    # Przygotuj dane
    klucz = "test_klucz"
    wynik = [{"test": "dane"}]
    
    # Wywołaj metodę
    rag_rynkowy._cache_wynik(klucz, wynik)
    
    # Sprawdź wynik
    assert klucz in rag_rynkowy._cache
    assert rag_rynkowy._cache[klucz]['wynik'] == wynik
    assert isinstance(rag_rynkowy._cache[klucz]['timestamp'], datetime)
    assert isinstance(rag_rynkowy._cache[klucz]['ttl'], timedelta)

def test_wyczysc_cache(rag_rynkowy):
    """Test czyszczenia wygasłych wpisów z cache'a."""
    # Przygotuj dane
    rag_rynkowy._cache = {
        'aktualny': {
            'wynik': [],
            'timestamp': datetime.now(),
            'ttl': timedelta(minutes=5)
        },
        'wygasly': {
            'wynik': [],
            'timestamp': datetime.now() - timedelta(minutes=10),
            'ttl': timedelta(minutes=5)
        }
    }
    
    # Wywołaj metodę
    rag_rynkowy._wyczysc_cache()
    
    # Sprawdź wynik
    assert 'aktualny' in rag_rynkowy._cache
    assert 'wygasly' not in rag_rynkowy._cache

@pytest.mark.asyncio
async def test_uruchom_automatyczne_aktualizacje(rag_rynkowy):
    """Test uruchamiania automatycznych aktualizacji."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    
    # Zamockuj asyncio.sleep
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        # Ustaw mock, żeby przerwał po pierwszej iteracji
        mock_sleep.side_effect = [None, Exception("Stop")]
        
        # Wywołaj metodę
        with pytest.raises(Exception, match="Stop"):
            await rag_rynkowy.uruchom_automatyczne_aktualizacje(dane, okno=10, interwal=1)
        
        # Sprawdź czy metody były wywoływane
        assert mock_sleep.called
        assert mock_sleep.call_args[0][0] == 1  # interwal

def test_znajdz_podobne_wzorce_z_cache(rag_rynkowy, rag_mock):
    """Test wyszukiwania wzorców z użyciem cache'a."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    
    # Pierwszy raz - bez cache'a
    wyniki1 = rag_rynkowy.znajdz_podobne_wzorce(dane, okno=10)
    assert len(wyniki1) == 1  # zgodnie z mockiem
    assert rag_mock.wyszukaj.call_count == 1
    
    # Drugi raz - powinien użyć cache'a
    wyniki2 = rag_rynkowy.znajdz_podobne_wzorce(dane, okno=10)
    assert len(wyniki2) == 1
    assert rag_mock.wyszukaj.call_count == 1  # nie powinno być nowego wywołania
    
    # Trzeci raz - z wyłączonym cache'm
    wyniki3 = rag_rynkowy.znajdz_podobne_wzorce(dane, okno=10, uzyj_cache=False)
    assert len(wyniki3) == 1
    assert rag_mock.wyszukaj.call_count == 2  # powinno być nowe wywołanie

def test_rozszerzony_opis_wzorca(rag_rynkowy):
    """Test generowania rozszerzonego opisu wzorca."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    
    # Wywołaj metodę
    opis = rag_rynkowy._przygotuj_opis_wzorca(dane, 20, okno=10)
    
    # Sprawdź wynik
    assert isinstance(opis, str)
    assert "Analiza cenowa:" in opis
    assert "Wskaźniki techniczne:" in opis
    assert "Analiza Wyckoffa:" in opis
    assert "RSI:" in opis
    assert "ATR:" in opis
    assert "Momentum:" in opis
    assert "Zmienność:" in opis
    assert "Względny wolumen:" in opis
    assert "Faza:" in opis 

def test_znajdz_podobne_wzorce_z_optymalizacja(rag_rynkowy, rag_mock):
    """Test wyszukiwania wzorców z parametrami optymalizacyjnymi."""
    # Przygotuj dane
    dane = generuj_dane_testowe(30)
    
    # Ustaw mock dla wyszukiwania
    rag_mock.wyszukaj.return_value = [
        {
            'tekst': 'Wzorzec 1',
            'metadata': {'data': '2024-01-01T00:00:00', 'zmiana_proc': 2.5},
            'score': 0.9
        },
        {
            'tekst': 'Wzorzec 2',
            'metadata': {'data': '2024-01-01T00:00:00', 'zmiana_proc': 1.5},
            'score': 0.8
        },
        {
            'tekst': 'Wzorzec 3',
            'metadata': {'data': '2024-01-01T00:00:00', 'zmiana_proc': 0.5},
            'score': 0.6
        }
    ]
    
    # Test progu podobieństwa
    wyniki = rag_rynkowy.znajdz_podobne_wzorce(
        dane,
        prog_podobienstwa=0.7,
        min_zmiana=1.0,
        max_zmiana=3.0
    )
    assert len(wyniki) == 2  # Tylko wzorce ze score >= 0.7 i zmianą w zakresie
    assert all(w['score'] >= 0.7 for w in wyniki)
    assert all(1.0 <= w['metadata']['zmiana_proc'] <= 3.0 for w in wyniki)
    
    # Test maksymalnego wieku
    wyniki = rag_rynkowy.znajdz_podobne_wzorce(
        dane,
        max_wiek_dni=7,
        prog_podobienstwa=0.0,  # Nie filtrujemy po podobieństwie
        n_wynikow=3  # Chcemy dokładnie tyle wyników ile jest w mock
    )
    assert len(wyniki) == 2  # Tylko dwa wzorce spełniają kryteria
    assert all(w['metadata']['data'] >= '2024-01-01T00:00:00' for w in wyniki)
    
    # Test kombinacji parametrów
    wyniki = rag_rynkowy.znajdz_podobne_wzorce(
        dane,
        prog_podobienstwa=0.85,
        max_wiek_dni=7,
        min_zmiana=2.0,
        n_wynikow=1
    )
    assert len(wyniki) == 1  # Tylko jeden wzorzec spełnia wszystkie kryteria
    assert wyniki[0]['score'] >= 0.85
    assert wyniki[0]['metadata']['zmiana_proc'] >= 2.0

def test_wydajnosc_duzych_zbiorow(rag_rynkowy, rag_mock):
    """Test wydajności dla dużych zbiorów danych."""
    # Przygotuj duży zbiór danych
    dane = generuj_dane_testowe(1000)
    
    # Przygotuj mock dla dużej liczby wyników
    duzo_wynikow = [
        {
            'tekst': f'Wzorzec {i}',
            'metadata': {
                'data': '2024-01-01T00:00:00',
                'zmiana_proc': 0.5 + (i / 10)  # Dodajemy zmianę procentową
            },
            'score': 0.5 + (i / 100)
        }
        for i in range(100)
    ]
    rag_mock.wyszukaj.return_value = duzo_wynikow
    
    # Test limitu wyników
    wyniki = rag_rynkowy.znajdz_podobne_wzorce(
        dane,
        n_wynikow=10,
        prog_podobienstwa=0.6
    )
    assert len(wyniki) == 10
    assert all(w['score'] >= 0.6 for w in wyniki)
    
    # Sprawdź sortowanie wyników
    scores = [w['score'] for w in wyniki]
    assert scores == sorted(scores, reverse=True) 