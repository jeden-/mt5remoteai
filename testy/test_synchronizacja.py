"""
Testy modułu synchronizacji danych z MT5.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import MetaTrader5 as mt5
from sqlalchemy.orm import Session
from unittest.mock import call

from baza_danych.synchronizacja import SynchronizatorMT5, TIMEFRAMES, SYMBOLS
from baza_danych.modele import (
    HistoriaCen, SynchronizacjaDanych, ZadanieAktualizacji,
    StatusSynchronizacji, StatusRynku
)
from baza_danych.cache import MenedzerCache

@pytest.fixture
def mock_session():
    """Fixture dostarczający zamockowaną sesję SQLAlchemy."""
    session = Mock(spec=Session)
    return session

@pytest.fixture
def mock_cache():
    """Fixture dostarczający zamockowany MenedzerCache."""
    return Mock(spec=MenedzerCache)

@pytest.fixture
def synchronizator(mock_session, mock_cache):
    """Fixture dostarczający instancję SynchronizatorMT5."""
    sync = SynchronizatorMT5(session=mock_session)
    sync.cache = mock_cache
    return sync

def test_dostepne_timeframes(synchronizator):
    """Test pobierania dostępnych timeframes."""
    timeframes = synchronizator.dostepne_timeframes()
    assert isinstance(timeframes, set)
    assert "M1" in timeframes
    assert "H1" in timeframes
    assert "D1" in timeframes
    assert len(timeframes) == len(TIMEFRAMES)

def test_dostepne_symbole(synchronizator):
    """Test pobierania dostępnych symboli."""
    symbole = synchronizator.dostepne_symbole()
    assert isinstance(symbole, set)
    assert "JP225" in symbole
    assert "USDJPY" in symbole
    assert len(symbole) == len(SYMBOLS)

@pytest.mark.asyncio
async def test_sprawdz_dostepnosc_symbolu_cache_hit(synchronizator, mock_cache):
    """Test sprawdzania dostępności symbolu - trafienie w cache."""
    mock_cache.pobierz.return_value = True
    
    assert await synchronizator.sprawdz_dostepnosc_symbolu("JP225") == True
    mock_cache.pobierz.assert_called_once()
    mock_cache.dodaj.assert_not_called()

@pytest.mark.asyncio
async def test_sprawdz_dostepnosc_symbolu_cache_miss(synchronizator, mock_cache):
    """Test sprawdzania dostępności symbolu - brak w cache."""
    mock_cache.pobierz.return_value = None
    
    with patch('MetaTrader5.symbol_info', return_value=Mock()):
        assert await synchronizator.sprawdz_dostepnosc_symbolu("JP225") == True
        mock_cache.pobierz.assert_called_once()
        mock_cache.dodaj.assert_called_once()

@pytest.mark.asyncio
async def test_sprawdz_dostepnosc_symbolu_nieobslugiwany(synchronizator, mock_cache):
    """Test sprawdzania dostępności symbolu - nieobsługiwany symbol."""
    mock_cache.pobierz.return_value = None
    
    assert await synchronizator.sprawdz_dostepnosc_symbolu("NIEZNANY") == False
    mock_cache.dodaj.assert_called_once_with(
        "symbol_dostepny_NIEZNANY",
        False,
        timedelta(hours=1)
    )

@pytest.mark.asyncio
async def test_synchronizuj_historie_cache_hit(synchronizator, mock_cache):
    """Test synchronizacji historii - trafienie w cache."""
    symbol = "JP225"
    timeframe = "M1"
    od = datetime.utcnow() - timedelta(hours=1)
    do = datetime.utcnow()
    
    mock_cache.pobierz.return_value = {
        'liczba_rekordow': 100,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    sukces = await synchronizator.synchronizuj_historie(symbol, timeframe, od, do)
    
    assert sukces == True
    assert mock_cache.pobierz.call_count == 2
    mock_cache.pobierz.assert_has_calls([
        call('symbol_dostepny_JP225'),
        call(f'historia_JP225_M1_{od.isoformat()}_{do.isoformat()}')
    ])

@pytest.mark.asyncio
async def test_synchronizuj_historie_cache_miss(synchronizator, mock_cache, mock_session):
    """Test synchronizacji historii - brak w cache."""
    symbol = "JP225"
    timeframe = "M1"
    od = datetime.utcnow() - timedelta(hours=1)
    do = datetime.utcnow()
    
    mock_cache.pobierz.side_effect = [True, None]  # symbol dostępny, brak historii
    mock_rates = [
        {
            'time': datetime.timestamp(od),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'tick_volume': 1000
        }
    ]
    
    with patch('MetaTrader5.symbol_info', return_value=Mock()), \
         patch('MetaTrader5.copy_rates_range', return_value=mock_rates):
        sukces = await synchronizator.synchronizuj_historie(symbol, timeframe, od, do)
        
        assert sukces == True
        assert mock_cache.pobierz.call_count == 2
        mock_cache.pobierz.assert_has_calls([
            call('symbol_dostepny_JP225'),
            call(f'historia_JP225_M1_{od.isoformat()}_{do.isoformat()}')
        ])

@pytest.mark.asyncio
async def test_aktualizuj_status_rynku_cache_hit(synchronizator, mock_cache):
    """Test aktualizacji statusu rynku - trafienie w cache."""
    symbol = "JP225"
    
    mock_cache.pobierz.return_value = {
        'otwarty': True,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    sukces = await synchronizator.aktualizuj_status_rynku(symbol)
    
    assert sukces == True
    mock_cache.pobierz.assert_called_once()
    mock_cache.dodaj.assert_not_called()

@pytest.mark.asyncio
async def test_aktualizuj_status_rynku_cache_miss(synchronizator, mock_cache, mock_session):
    """Test aktualizacji statusu rynku - brak w cache."""
    symbol = "JP225"
    mock_info = Mock(trade_mode=3)  # 3 = pełny dostęp do handlu

    mock_cache.pobierz.side_effect = [None, True]  # brak statusu, symbol dostępny

    with patch('MetaTrader5.symbol_info', return_value=mock_info):
        sukces = await synchronizator.aktualizuj_status_rynku(symbol)

        assert sukces == True
        assert mock_cache.pobierz.call_count == 2
        mock_cache.pobierz.assert_has_calls([
            call('status_rynku_JP225'),
            call('symbol_dostepny_JP225')
        ], any_order=False)

@pytest.mark.asyncio
async def test_przetworz_zadania(synchronizator, mock_session):
    """Test przetwarzania kolejki zadań."""
    
    # Przygotowanie danych testowych
    zadania = [
        ZadanieAktualizacji(
            symbol="JP225",
            timeframe="M1",
            typ="historia_cen",
            priorytet=1
        ),
        ZadanieAktualizacji(
            symbol="JP225",
            timeframe="M1",
            typ="status",
            priorytet=2
        )
    ]
    
    # Mock zapytania do bazy
    mock_session.execute().scalars.return_value.all.return_value = zadania
    
    # Mock metod synchronizacji
    with patch.object(synchronizator, 'synchronizuj_historie', return_value=True), \
         patch.object(synchronizator, 'aktualizuj_status_rynku', return_value=True):
        
        # Wywołanie testowanej metody
        await synchronizator.przetworz_zadania()
        
        # Sprawdzenie rezultatów
        assert all(zadanie.wykonane for zadanie in zadania)
        assert mock_session.commit.called

@pytest.mark.asyncio
async def test_przetworz_zadania_blad_zadania(synchronizator, mock_session):
    """Test błędu podczas przetwarzania pojedynczego zadania."""
    
    zadanie = ZadanieAktualizacji(
        symbol="JP225",
        timeframe="M1",
        typ="historia_cen",
        priorytet=1
    )
    
    mock_session.execute().scalars.return_value.all.return_value = [zadanie]
    
    with patch.object(synchronizator, 'synchronizuj_historie', side_effect=Exception("Test error")):
        await synchronizator.przetworz_zadania()
        
        assert zadanie.wykonane == False
        assert zadanie.blad == "Test error"
        assert mock_session.commit.called

@pytest.mark.asyncio
async def test_przetworz_zadania_nieznany_typ(synchronizator, mock_session):
    """Test przetwarzania zadania o nieznanym typie."""
    
    zadanie = ZadanieAktualizacji(
        symbol="JP225",
        timeframe="M1",
        typ="nieznany_typ",
        priorytet=1
    )
    
    mock_session.execute().scalars.return_value.all.return_value = [zadanie]
    
    await synchronizator.przetworz_zadania()
    
    assert zadanie.wykonane == False
    assert zadanie.blad == "Nie udało się wykonać zadania"
    assert mock_session.commit.called

@pytest.mark.asyncio
async def test_synchronizuj_historie_blad_mt5(synchronizator, mock_cache):
    """Test synchronizacji historii - błąd MT5."""
    symbol = "JP225"
    timeframe = "M1"
    od = datetime.utcnow() - timedelta(hours=1)
    do = datetime.utcnow()

    mock_cache.pobierz.side_effect = [True, None]  # symbol dostępny, brak historii

    with patch('MetaTrader5.copy_rates_range', return_value=None):
        sukces = await synchronizator.synchronizuj_historie(symbol, timeframe, od, do)
        assert not sukces

@pytest.mark.asyncio
async def test_synchronizuj_historie_blad_zapisu(synchronizator, mock_cache, mock_session):
    """Test synchronizacji historii - błąd zapisu do bazy."""
    symbol = "JP225"
    timeframe = "M1"
    od = datetime.utcnow() - timedelta(hours=1)
    do = datetime.utcnow()

    mock_cache.pobierz.side_effect = [True, None]  # symbol dostępny, brak historii
    mock_rates = [
        {
            'time': datetime.timestamp(od),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'tick_volume': 1000
        }
    ]

    # Pierwszy commit() przejdzie, drugi rzuci wyjątek, trzeci przejdzie (przy obsłudze błędu)
    mock_session.commit.side_effect = [None, Exception("Błąd zapisu"), None]

    with patch('MetaTrader5.copy_rates_range', return_value=mock_rates):
        sukces = await synchronizator.synchronizuj_historie(symbol, timeframe, od, do)
        assert not sukces
        assert mock_session.commit.call_count == 3  # Pierwszy commit, próba drugiego i commit przy obsłudze błędu

@pytest.mark.asyncio
async def test_aktualizuj_status_rynku_blad_mt5(synchronizator, mock_cache):
    """Test aktualizacji statusu rynku - błąd MT5."""
    symbol = "JP225"

    mock_cache.pobierz.side_effect = [None, True]  # brak statusu, symbol dostępny

    with patch('MetaTrader5.symbol_info', return_value=None):
        sukces = await synchronizator.aktualizuj_status_rynku(symbol)
        assert not sukces

@pytest.mark.asyncio
async def test_przetworz_zadania_blad_sesji(synchronizator, mock_session):
    """Test przetwarzania zadań - błąd sesji."""
    mock_session.execute.side_effect = Exception("Błąd sesji")

    await synchronizator.przetworz_zadania()  # Nie powinno rzucić wyjątku 