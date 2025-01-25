"""
Testy modułu cache.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from baza_danych.cache import MenedzerCache
from baza_danych.modele import Cache

@pytest.fixture
def mock_session():
    """Fixture dostarczający zamockowaną sesję SQLAlchemy."""
    session = Mock(spec=Session)
    return session

@pytest.fixture
def menedzer_cache(mock_session):
    """Fixture dostarczający instancję MenedzerCache."""
    return MenedzerCache(session=mock_session)

@pytest.mark.asyncio
async def test_dodaj_cache(menedzer_cache, mock_session):
    """Test dodawania wartości do cache."""
    
    # Przygotowanie danych testowych
    klucz = "test_klucz"
    wartosc = {"test": "wartosc"}
    czas_wygasniecia = timedelta(hours=1)
    
    # Wywołanie testowanej metody
    sukces = await menedzer_cache.dodaj(klucz, wartosc, czas_wygasniecia)
    
    # Sprawdzenie rezultatów
    assert sukces == True
    assert mock_session.add.called
    assert mock_session.commit.called
    
    # Sprawdzenie utworzonego wpisu
    cache_wpis = mock_session.add.call_args[0][0]
    assert isinstance(cache_wpis, Cache)
    assert cache_wpis.klucz == klucz
    assert json.loads(cache_wpis.wartosc) == wartosc
    assert cache_wpis.wygasa is not None

@pytest.mark.asyncio
async def test_dodaj_cache_bez_wygasniecia(menedzer_cache, mock_session):
    """Test dodawania wartości do cache bez czasu wygaśnięcia."""
    
    klucz = "test_klucz"
    wartosc = {"test": "wartosc"}
    
    sukces = await menedzer_cache.dodaj(klucz, wartosc)
    
    assert sukces == True
    cache_wpis = mock_session.add.call_args[0][0]
    assert cache_wpis.wygasa is None

@pytest.mark.asyncio
async def test_dodaj_cache_blad(menedzer_cache, mock_session):
    """Test błędu podczas dodawania do cache."""
    
    mock_session.commit.side_effect = SQLAlchemyError("Test error")
    
    sukces = await menedzer_cache.dodaj("test", "wartosc")
    
    assert sukces == False

@pytest.mark.asyncio
async def test_pobierz_cache_istniejacy(menedzer_cache, mock_session):
    """Test pobierania istniejącej wartości z cache."""
    
    # Przygotowanie danych testowych
    klucz = "test_klucz"
    wartosc = {"test": "wartosc"}
    mock_cache = Mock(
        klucz=klucz,
        wartosc=json.dumps(wartosc),
        wygasa=datetime.utcnow() + timedelta(hours=1)
    )
    
    # Mock zapytania do bazy
    mock_session.execute().scalar_one_or_none.return_value = mock_cache
    
    # Wywołanie testowanej metody
    wynik = await menedzer_cache.pobierz(klucz)
    
    # Sprawdzenie rezultatów
    assert wynik == wartosc

@pytest.mark.asyncio
async def test_pobierz_cache_nieistniejacy(menedzer_cache, mock_session):
    """Test pobierania nieistniejącej wartości z cache."""
    
    mock_session.execute().scalar_one_or_none.return_value = None
    
    wynik = await menedzer_cache.pobierz("nieistniejacy_klucz")
    
    assert wynik is None

@pytest.mark.asyncio
async def test_pobierz_cache_blad(menedzer_cache, mock_session):
    """Test błędu podczas pobierania z cache."""
    
    mock_session.execute.side_effect = SQLAlchemyError("Test error")
    
    wynik = await menedzer_cache.pobierz("test")
    
    assert wynik is None

@pytest.mark.asyncio
async def test_usun_cache(menedzer_cache, mock_session):
    """Test usuwania wartości z cache."""
    
    klucz = "test_klucz"
    
    sukces = await menedzer_cache.usun(klucz)
    
    assert sukces == True
    assert mock_session.execute.called
    assert mock_session.commit.called

@pytest.mark.asyncio
async def test_usun_cache_blad(menedzer_cache, mock_session):
    """Test błędu podczas usuwania z cache."""
    
    mock_session.execute.side_effect = SQLAlchemyError("Test error")
    
    sukces = await menedzer_cache.usun("test")
    
    assert sukces == False

@pytest.mark.asyncio
async def test_wyczysc_wygasle(menedzer_cache, mock_session):
    """Test czyszczenia wygasłych wpisów z cache."""
    
    # Mock wyniku zapytania
    mock_session.execute().rowcount = 5
    
    # Wywołanie testowanej metody
    liczba = await menedzer_cache.wyczysc_wygasle()
    
    # Sprawdzenie rezultatów
    assert liczba == 5
    assert mock_session.execute.called
    assert mock_session.commit.called

@pytest.mark.asyncio
async def test_wyczysc_wygasle_blad(menedzer_cache, mock_session):
    """Test błędu podczas czyszczenia wygasłych wpisów."""
    
    mock_session.execute.side_effect = SQLAlchemyError("Test error")
    
    liczba = await menedzer_cache.wyczysc_wygasle()
    
    assert liczba == 0

@pytest.mark.asyncio
async def test_pobierz_wiele_cache(menedzer_cache, mock_session):
    """Test pobierania wielu wartości z cache."""
    
    # Przygotowanie danych testowych
    klucze = ["klucz1", "klucz2"]
    wartosci = {
        "klucz1": {"test1": "wartosc1"},
        "klucz2": {"test2": "wartosc2"}
    }
    mock_wpisy = [
        Mock(
            klucz=k,
            wartosc=json.dumps(v),
            wygasa=datetime.utcnow() + timedelta(hours=1)
        )
        for k, v in wartosci.items()
    ]
    
    # Mock zapytania do bazy
    mock_session.execute().scalars.return_value.all.return_value = mock_wpisy
    
    # Wywołanie testowanej metody
    wynik = await menedzer_cache.pobierz_wiele(klucze)
    
    # Sprawdzenie rezultatów
    assert wynik == wartosci

@pytest.mark.asyncio
async def test_pobierz_wiele_cache_blad_deserializacji(menedzer_cache, mock_session):
    """Test obsługi błędu deserializacji przy pobieraniu wielu wartości."""
    
    klucze = ["klucz1", "klucz2"]
    mock_wpisy = [
        Mock(klucz="klucz1", wartosc="nieprawidlowy_json"),
        Mock(klucz="klucz2", wartosc=json.dumps({"test": "ok"}))
    ]
    
    mock_session.execute().scalars.return_value.all.return_value = mock_wpisy
    
    wynik = await menedzer_cache.pobierz_wiele(klucze)
    
    assert len(wynik) == 1
    assert "klucz2" in wynik
    assert wynik["klucz2"] == {"test": "ok"}

@pytest.mark.asyncio
async def test_pobierz_wiele_cache_blad_zapytania(menedzer_cache, mock_session):
    """Test błędu zapytania przy pobieraniu wielu wartości."""
    
    mock_session.execute.side_effect = SQLAlchemyError("Test error")
    
    wynik = await menedzer_cache.pobierz_wiele(["test1", "test2"])
    
    assert wynik == {} 