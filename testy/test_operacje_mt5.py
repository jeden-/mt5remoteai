"""
Testy dla modułu operacje_mt5.py
"""
import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
from datetime import datetime
from freezegun import freeze_time
import MetaTrader5 as mt5
from handel.operacje_mt5 import OperacjeHandloweMT5

@pytest.fixture
def operacje():
    return OperacjeHandloweMT5()

@pytest.fixture
def mock_mt5(mocker):
    mt5_mock = mocker.patch('handel.operacje_mt5.mt5')
    mt5_mock.initialize.return_value = True
    mt5_mock.terminal_info.return_value = MagicMock(trade_allowed=True)
    return mt5_mock

def test_inicjalizacja_blad_salda(operacje, mock_mt5):
    """Test inicjalizacji gdy nie można pobrać salda konta."""
    mock_mt5.account_info.return_value = None
    
    assert operacje.inicjalizuj() == True

def test_pobierz_dane_historyczne_filtrowanie_daty(operacje, mock_mt5):
    """Test filtrowania danych historycznych po dacie końcowej."""
    dane = pd.DataFrame({
        'time': pd.date_range(start='2023-01-01', end='2023-01-10', freq='D'),
        'open': [1.0] * 10,
        'high': [1.1] * 10,
        'low': [0.9] * 10,
        'close': [1.0] * 10,
        'tick_volume': [100] * 10,
        'spread': [2] * 10,
        'real_volume': [1000] * 10
    })
    mock_mt5.copy_rates_from_pos.return_value = dane.values
    
    operacje.inicjalizuj()
    
    end_date = pd.Timestamp('2023-01-05')
    wynik = operacje.pobierz_dane_historyczne('EURUSD', mock_mt5.TIMEFRAME_D1, 10, end_date=end_date)
    
    assert len(wynik) == 5
    assert wynik.index[-1].date() == end_date.date()

@pytest.mark.parametrize("typ,dane", [
    ('wiadomosc', {'tytul': 'Test', 'tresc': 'Testowa wiadomość'}),
    ('sesja', {'symbol': 'EURUSD', 'start': '09:00', 'koniec': '17:00'}),
    ('status', {'symbol': 'EURUSD', 'otwarty': True, 'powod': None}),
    ('aktyw', {'nazwa': 'EURUSD', 'opis': 'Euro/USD', 'spread': 1}),
    ('metryki', {'symbol': 'EURUSD', 'wolumen': 1.0, 'zysk': 100}),
    ('cena', {'symbol': 'EURUSD', 'cena': 1.1234, 'timestamp': '2024-01-01'})
])
def test_zapis_do_bazy_rozne_typy(mocker, operacje, typ, dane):
    """Test zapisywania różnych typów danych do bazy."""
    mock_baza = mocker.MagicMock()
    operacje.baza = mock_baza
    
    operacje._zapisz_do_bazy(dane, typ)
    
    if typ == 'wiadomosc':
        mock_baza.dodaj_wiadomosc.assert_called_once_with(dane)
    elif typ == 'sesja':
        mock_baza.dodaj_sesje_handlowa.assert_called_once_with(dane)
    elif typ == 'status':
        mock_baza.aktualizuj_status_rynku.assert_called_once_with(dane['symbol'], dane)
    elif typ == 'aktyw':
        mock_baza.aktualizuj_aktyw.assert_called_once_with(dane['nazwa'], dane)
    elif typ == 'metryki':
        mock_baza.dodaj_metryki.assert_called_once_with(dane)
    elif typ == 'cena':
        mock_baza.dodaj_cene.assert_called_once_with(dane)

def test_zapis_do_bazy_brak_bazy(operacje):
    """Test próby zapisu gdy baza nie jest skonfigurowana."""
    operacje.baza = None
    
    # Nie powinno rzucić wyjątku
    operacje._zapisz_do_bazy({'test': 'dane'}, 'wiadomosc')

def test_zapis_do_bazy_blad_zapisu(mocker, operacje):
    """Test błędu podczas zapisu do bazy."""
    mock_baza = mocker.MagicMock()
    mock_baza.dodaj_wiadomosc.side_effect = Exception("Błąd zapisu")
    operacje.baza = mock_baza
    
    # Nie powinno rzucić wyjątku
    operacje._zapisz_do_bazy({'test': 'dane'}, 'wiadomosc')

def test_zapis_do_bazy_nieprawidlowy_typ(mocker, operacje):
    """Test zapisu z nieprawidłowym typem danych."""
    mock_baza = mocker.MagicMock()
    operacje.baza = mock_baza
    
    operacje._zapisz_do_bazy({'test': 'dane'}, 'nieprawidlowy_typ')
    
    # Żadna metoda nie powinna być wywołana
    assert not mock_baza.method_calls

@freeze_time("2023-01-01 00:00:00")
def test_sprawdz_status_rynku_przesuniecie_daty(operacje, mock_mt5):
    """Test przesunięcia daty otwarcia rynku o jeden dzień."""
    mock_mt5.symbol_info.return_value = MagicMock(
        session_deals=0,
        session_buy_orders=0,
        session_sell_orders=0,
        volume=0,
        volumehigh=0,
        volumelow=0
    )
    
    operacje.inicjalizuj()
    
    status = operacje.sprawdz_status_rynku("EURUSD")
    assert status["nastepne_otwarcie"].date() == datetime(2023, 1, 2).date()

def test_pobierz_kalendarz_brak_wartosci(operacje, mock_mt5):
    """Test pobierania kalendarza gdy brak wartości dla wydarzeń."""
    mock_mt5.calendar_get.return_value = None
    
    operacje.inicjalizuj()
    
    kalendarz = operacje.pobierz_kalendarz()
    assert kalendarz is None

def test_pobierz_wiadomosci_blad_ogolny(operacje, mock_mt5):
    """Test pobierania wiadomości gdy wystąpi ogólny błąd."""
    mock_mt5.news_get.side_effect = Exception("Test error")
    
    operacje.inicjalizuj()
    
    wiadomosci = operacje.pobierz_wiadomosci()
    assert wiadomosci is None

def test_pobierz_historie_konta_blad_ogolny(operacje, mock_mt5):
    """Test pobierania historii konta gdy wystąpi ogólny błąd."""
    mock_mt5.history_deals_get.side_effect = Exception("Test error")
    
    operacje.inicjalizuj()
    
    historia = operacje.pobierz_historie_konta()
    assert historia is None

def test_pobierz_aktywa_blad_ogolny(operacje, mock_mt5):
    """Test pobierania aktywów gdy wystąpi ogólny błąd."""
    mock_mt5.symbols_get.side_effect = Exception("Test error")
    
    operacje.inicjalizuj()
    
    aktywa = operacje.pobierz_aktywa()
    assert aktywa is None

def test_pobierz_swieta_blad_ogolny(operacje, mock_mt5):
    """Test pobierania świąt gdy wystąpi ogólny błąd."""
    mock_mt5.symbol_info_get.side_effect = Exception("Test error")
    
    operacje.inicjalizuj()
    
    swieta = operacje.pobierz_swieta("EURUSD")
    assert swieta is None

def test_sprawdz_status_rynku_blad_tick_info(operacje, mock_mt5):
    """Test sprawdzania statusu rynku gdy wystąpi błąd przy pobieraniu informacji o ticku."""
    mock_mt5.symbol_info.return_value = MagicMock(trade_mode=True)
    mock_mt5.symbol_info_tick.side_effect = Exception("Test error")
    
    operacje.inicjalizuj()
    
    status = operacje.sprawdz_status_rynku("EURUSD")
    assert status["otwarty"] == False
    assert status["powod"] == "Brak informacji o sesjach"
    assert status["nastepne_otwarcie"] is None

def test_otworz_pozycje_blad_deviation(operacje, mock_mt5):
    """Test otwierania pozycji gdy cena zmieni się poza dozwolone odchylenie."""
    mock_mt5.symbol_info.return_value = MagicMock(
        ask=1.1000,
        bid=1.0998
    )
    
    result = Mock()
    result.retcode = mt5.TRADE_RETCODE_REQUOTE
    result.comment = "Requote"
    mock_mt5.order_send.return_value = result
    
    operacje.inicjalizuj()
    
    ticket = operacje.otworz_pozycje("EURUSD", "BUY", 0.1)
    assert ticket is None

def test_sprawdz_status_rynku_blad_timezone(operacje, mock_mt5):
    """Test sprawdzania statusu rynku gdy wystąpi błąd z strefą czasową."""
    mock_mt5.symbol_info.return_value = MagicMock(
        trade_mode=True,
        session_deals=True
    )
    mock_mt5.symbol_info_tick.return_value = Mock()
    
    operacje.inicjalizuj()
    
    # Podmiana strefy czasowej na None
    original_timezone = operacje.timezone
    operacje.timezone = None
    
    status = operacje.sprawdz_status_rynku("EURUSD")
    assert status["otwarty"] == False
    assert status["powod"] == "Błąd konfiguracji strefy czasowej"
    
    # Przywrócenie oryginalnej strefy czasowej
    operacje.timezone = original_timezone

def test_otworz_pozycje_nieprawidlowy_wolumen(operacje, mock_mt5):
    """Test otwierania pozycji z nieprawidłowym wolumenem."""
    mock_mt5.terminal_info.return_value = MagicMock(trade_allowed=True)
    mock_mt5.account_info.return_value = MagicMock(balance=1000.0)
    mock_mt5.symbols_get.return_value = [MagicMock(name="EURUSD")]
    mock_mt5.symbol_info.return_value = MagicMock(ask=1.1000, bid=1.0998)
    
    operacje.inicjalizuj()
    
    # Test dla wolumenu 0
    ticket = operacje.otworz_pozycje("EURUSD", "BUY", 0)
    assert ticket is None
    
    # Test dla ujemnego wolumenu
    ticket = operacje.otworz_pozycje("EURUSD", "BUY", -1)
    assert ticket is None

def test_otworz_pozycje_nieistniejacy_symbol(operacje, mock_mt5):
    """Test otwierania pozycji dla nieistniejącego symbolu."""
    mock_mt5.terminal_info.return_value = MagicMock(trade_allowed=True)
    mock_mt5.symbols_get.return_value = [MagicMock(name="EURUSD")]
    
    operacje.inicjalizuj()
    
    ticket = operacje.otworz_pozycje("INVALID", "BUY", 0.1)
    assert ticket is None

def test_otworz_pozycje_nieprawidlowe_sl_tp(operacje, mock_mt5):
    """Test otwierania pozycji z nieprawidłowymi poziomami SL/TP."""
    mock_mt5.terminal_info.return_value = MagicMock(trade_allowed=True)
    mock_mt5.account_info.return_value = MagicMock(balance=1000.0)
    mock_mt5.symbols_get.return_value = [MagicMock(name="EURUSD")]
    mock_mt5.symbol_info.return_value = MagicMock(ask=1.1000, bid=1.0998)
    
    operacje.inicjalizuj()
    
    # Test dla BUY z SL powyżej ceny wejścia
    ticket = operacje.otworz_pozycje("EURUSD", "BUY", 0.1, sl=1.1500, tp=1.1200)
    assert ticket is None
    
    # Test dla SELL z SL poniżej ceny wejścia
    ticket = operacje.otworz_pozycje("EURUSD", "SELL", 0.1, sl=1.0500, tp=1.0800)
    assert ticket is None

def test_otworz_pozycje_brak_srodkow(operacje, mock_mt5):
    """Test otwierania pozycji gdy brak wystarczających środków."""
    mock_mt5.terminal_info.return_value = MagicMock(trade_allowed=True)
    mock_mt5.account_info.return_value = MagicMock(balance=10.0)  # Małe saldo
    mock_mt5.symbols_get.return_value = [MagicMock(name="EURUSD")]
    mock_mt5.symbol_info.return_value = MagicMock(ask=1.1000, bid=1.0998)
    
    operacje.inicjalizuj()
    
    ticket = operacje.otworz_pozycje("EURUSD", "BUY", 1.0)  # Duży wolumen
    assert ticket is None

def test_zamknij_pozycje_nieistniejaca(operacje, mock_mt5):
    """Test zamykania nieistniejącej pozycji."""
    mock_mt5.positions_get.return_value = None
    
    operacje.inicjalizuj()
    
    assert operacje.zamknij_pozycje(12345) == False

def test_zamknij_pozycje_blad_zlecenia(operacje, mock_mt5):
    """Test błędu podczas wysyłania zlecenia zamknięcia pozycji."""
    pozycja = MagicMock(
        symbol="EURUSD",
        type=mt5.ORDER_TYPE_BUY,
        volume=0.1,
        ticket=12345
    )
    mock_mt5.positions_get.return_value = [pozycja]
    
    result = Mock()
    result.retcode = mt5.TRADE_RETCODE_ERROR
    result.comment = "Test error"
    mock_mt5.order_send.return_value = result
    
    mock_mt5.symbol_info_tick.return_value = MagicMock(bid=1.1000, ask=1.0998)
    
    operacje.inicjalizuj()
    
    assert operacje.zamknij_pozycje(12345) == False

def test_zamknij_pozycje_brak_inicjalizacji(operacje, mock_mt5):
    """Test zamykania pozycji gdy MT5 nie jest zainicjalizowany."""
    # Nie wywołujemy inicjalizuj()
    assert operacje.zamknij_pozycje(12345) == False

def test_zamknij_pozycje_blad_info(operacje, mock_mt5):
    """Test błędu podczas pobierania informacji o pozycji."""
    mock_mt5.positions_get.side_effect = Exception("Test error")
    
    operacje.inicjalizuj()
    
    assert operacje.zamknij_pozycje(12345) == False

def test_otworz_pozycje_brak_inicjalizacji(operacje, mock_mt5):
    """Test otwierania pozycji gdy MT5 nie jest zainicjalizowany."""
    # Nie wywołujemy inicjalizuj()
    ticket = operacje.otworz_pozycje("EURUSD", "BUY", 0.1)
    assert ticket is None

def test_otworz_pozycje_handel_wylaczony(operacje, mock_mt5):
    """Test otwierania pozycji gdy handel jest wyłączony."""
    mock_mt5.terminal_info.return_value = MagicMock(trade_allowed=False)
    
    operacje.inicjalizuj()
    ticket = operacje.otworz_pozycje("EURUSD", "BUY", 0.1)
    assert ticket is None

def test_otworz_pozycje_nieprawidlowy_typ(operacje, mock_mt5):
    """Test otwierania pozycji z nieprawidłowym typem zlecenia."""
    mock_mt5.terminal_info.return_value = MagicMock(trade_allowed=True)
    
    operacje.inicjalizuj()
    ticket = operacje.otworz_pozycje("EURUSD", "INVALID", 0.1)
    assert ticket is None

def test_otworz_pozycje_blad_terminal_info(operacje, mock_mt5):
    """Test otwierania pozycji gdy nie można pobrać informacji o terminalu."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = None
    mock_mt5.account_info.return_value = MagicMock(balance=1000)
    
    assert not operacje.inicjalizuj()
    mock_mt5.shutdown.assert_called_once()

def test_inicjalizacja_blad_inicjalizacji(operacje, mock_mt5):
    """Test inicjalizacji gdy MT5 nie może się zainicjalizować."""
    mock_mt5.initialize.return_value = False
    
    assert not operacje.inicjalizuj()
    mock_mt5.shutdown.assert_called_once()

def test_inicjalizacja_blad_ogolny(operacje, mock_mt5):
    """Test inicjalizacji gdy wystąpi ogólny błąd."""
    mock_mt5.initialize.side_effect = Exception("Błąd ogólny")
    
    assert not operacje.inicjalizuj()
    mock_mt5.shutdown.assert_called_once()

def test_pobierz_nastepne_otwarcie_weekend(operacje, mock_mt5):
    """Test pobierania następnego otwarcia rynku w weekend."""
    with freeze_time("2024-01-06 12:00:00"):  # Sobota
        next_open = operacje.pobierz_nastepne_otwarcie("EURUSD")
        assert next_open.weekday() == 0  # Poniedziałek
        assert next_open.hour == 8
        assert next_open.minute == 0

def test_pobierz_nastepne_otwarcie_dzien_roboczy(operacje, mock_mt5):
    """Test pobierania następnego otwarcia rynku w dzień roboczy."""
    with freeze_time("2024-01-02 12:00:00"):  # Wtorek
        next_open = operacje.pobierz_nastepne_otwarcie("EURUSD")
        assert next_open.date() == datetime(2024, 1, 3).date()  # Środa
        assert next_open.hour == 8
        assert next_open.minute == 0

def test_sprawdz_status_rynku_blad_symbol_info(operacje, mock_mt5):
    """Test sprawdzania statusu rynku gdy nie można pobrać informacji o symbolu."""
    mock_mt5.symbol_info.return_value = None
    
    operacje.inicjalizuj()
    status = operacje.sprawdz_status_rynku("EURUSD")
    
    assert status["otwarty"] == False
    assert status["powod"] == "Symbol nie istnieje"
    assert status["nastepne_otwarcie"] is None

def test_sprawdz_status_rynku_blad_trade_mode(operacje, mock_mt5):
    """Test sprawdzania statusu rynku gdy symbol nie jest dostępny do handlu."""
    mock_mt5.symbol_info.return_value = MagicMock(trade_mode=0)  # Handel wyłączony
    
    operacje.inicjalizuj()
    status = operacje.sprawdz_status_rynku("EURUSD")
    
    assert status["otwarty"] == False
    assert status["powod"] == "Symbol niedostępny do handlu"
    assert status["nastepne_otwarcie"] is not None

def test_sprawdz_status_rynku_blad_ogolny(operacje, mock_mt5):
    """Test sprawdzania statusu rynku gdy wystąpi ogólny błąd."""
    mock_mt5.symbol_info.side_effect = Exception("Błąd ogólny")
    
    operacje.inicjalizuj()
    status = operacje.sprawdz_status_rynku("EURUSD")
    
    assert status["otwarty"] == False
    assert status["powod"] == "Błąd podczas sprawdzania statusu"
    assert status["nastepne_otwarcie"] is None