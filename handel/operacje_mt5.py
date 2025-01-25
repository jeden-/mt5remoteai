"""
Moduł odpowiedzialny za operacje handlowe i pobieranie danych z platformy MT5.
Zawiera funkcje do handlu, pobierania danych historycznych i kalendarza ekonomicznego.
"""

import MetaTrader5 as mt5
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import pytz
from baza_danych.baza import BazaDanych
from unittest.mock import MagicMock
from pytz import timezone

# Konfiguracja loggera
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class OperacjeHandloweMT5:
    """Klasa do zarządzania operacjami handlowymi i danymi z MT5."""
    
    def __init__(self, baza: Optional[BazaDanych] = None):
        """
        Inicjalizacja obiektu do operacji handlowych.
        
        Args:
            baza: Opcjonalne połączenie z bazą danych
        """
        self.logger = logging.getLogger(__name__)
        self.initialized = False
        self.symbols = None
        self.account_info = None
        self.timezone = pytz.timezone('Europe/Warsaw')
        self.baza = baza
    
    def inicjalizuj(self) -> bool:
        """Inicjalizuje połączenie z MT5."""
        try:
            if not mt5.initialize():
                self.logger.error("❌ Nie udało się zainicjalizować MT5")
                mt5.shutdown()
                return False

            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.logger.error("❌ Nie udało się pobrać informacji o terminalu")
                mt5.shutdown()
                return False

            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("❌ Nie udało się pobrać informacji o koncie")
                self.account_info = {}
            else:
                self.account_info = account_info

            self.symbols = mt5.symbols_get()
            if self.symbols is None:
                self.logger.error("❌ Nie udało się pobrać listy symboli")
                self.symbols = []

            self.initialized = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Błąd podczas inicjalizacji MT5: {e}")
            mt5.shutdown()
            return False
    
    def pobierz_dostepne_symbole(self) -> List[str]:
        """
        Zwraca listę dostępnych symboli.
        
        Returns:
            Lista nazw dostępnych symboli
        """
        if not self.initialized:
            logger.error("❌ MT5 nie jest zainicjalizowany")
            return []
            
        return [symbol.name for symbol in self.symbols]
    
    def pobierz_nastepne_otwarcie(self, symbol: str) -> Optional[datetime]:
        """Pobiera następną datę otwarcia rynku dla danego symbolu."""
        try:
            now = datetime.now(tz=timezone('Europe/Warsaw'))
            
            # Jeśli weekend, przesuwamy na poniedziałek
            if now.weekday() >= 5:  # Sobota lub niedziela
                next_open = now + timedelta(days=(7 - now.weekday()))  # Przesuwamy na poniedziałek
                next_open = next_open.replace(hour=8, minute=0, second=0, microsecond=0)
            else:
                # Dla dni roboczych
                next_open = now + timedelta(days=1)  # Następny dzień
                next_open = next_open.replace(hour=8, minute=0, second=0, microsecond=0)
                
            return next_open
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas pobierania następnego otwarcia: {e}")
            return None

    def sprawdz_status_rynku(self, symbol: str) -> Dict[str, Any]:
        """Sprawdza status rynku dla danego symbolu."""
        if not self.initialized:
            return {
                "otwarty": False,
                "powod": "MT5 nie jest zainicjalizowany",
                "nastepne_otwarcie": None
            }

        if not mt5.timezone() or self.timezone is None:
            return {
                "otwarty": False,
                "powod": "Błąd konfiguracji strefy czasowej",
                "nastepne_otwarcie": None
            }

        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                self.logger.error(f"❌ Nie znaleziono symbolu {symbol}")
                return {
                    "otwarty": False,
                    "powod": "Symbol nie istnieje",
                    "nastepne_otwarcie": None
                }

            if not info.trade_mode:
                self.logger.error(f"❌ Symbol {symbol} nie jest dostępny do handlu")
                nastepne_otwarcie = self.pobierz_nastepne_otwarcie(symbol)
                return {
                    "otwarty": False,
                    "powod": "Symbol niedostępny do handlu",
                    "nastepne_otwarcie": nastepne_otwarcie
                }

            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    self.logger.error(f"❌ Nie udało się pobrać informacji o ticku dla {symbol}")
                    return {
                        "otwarty": False,
                        "powod": "Brak informacji o sesjach",
                        "nastepne_otwarcie": None
                    }
            except Exception as e:
                self.logger.error(f"❌ Błąd podczas pobierania informacji o ticku: {e}")
                return {
                    "otwarty": False,
                    "powod": "Brak informacji o sesjach",
                    "nastepne_otwarcie": None
                }

            # Sprawdź czy rynek jest zamknięty
            if (info.session_deals == 0 and info.session_buy_orders == 0 
                and info.session_sell_orders == 0 and info.volume == 0 
                and info.volumehigh == 0 and info.volumelow == 0):
                
                nastepne_otwarcie = self.pobierz_nastepne_otwarcie(symbol)
                return {
                    "otwarty": False,
                    "powod": "Rynek zamknięty",
                    "nastepne_otwarcie": nastepne_otwarcie
                }

            return {
                "otwarty": True,
                "powod": None,
                "nastepne_otwarcie": None
            }

        except Exception as e:
            self.logger.error(f"❌ Błąd podczas sprawdzania statusu rynku: {e}")
            return {
                "otwarty": False,
                "powod": "Błąd podczas sprawdzania statusu",
                "nastepne_otwarcie": None
            }
    
    def pobierz_dane_historyczne(self, symbol: str, timeframe: int, count: int, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Pobiera dane historyczne dla danego symbolu."""
        if not isinstance(timeframe, (int, MagicMock)):
            self.logger.error(f"❌ Nieprawidłowy timeframe: {timeframe}")
            return None

        try:
            if start_date:
                rates = mt5.copy_rates_from(symbol, timeframe, start_date, count)
            else:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

            if rates is None:
                self.logger.error(f"❌ Nie udało się pobrać danych historycznych dla {symbol}")
                return None

            df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            if end_date:
                df = df[df.index <= end_date]

            return df

        except Exception as e:
            self.logger.error(f"❌ Błąd podczas pobierania danych historycznych: {e}")
            return None
    
    def otworz_pozycje(
        self,
        symbol: str,
        typ: str,
        wolumen: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        komentarz: str = ""
    ) -> Optional[int]:
        """
        Otwiera nową pozycję handlową.
        
        Args:
            symbol: Symbol (np. "EURUSD")
            typ: Typ zlecenia ("BUY" lub "SELL")
            wolumen: Wielkość pozycji w lotach
            sl: Poziom Stop Loss (opcjonalny)
            tp: Poziom Take Profit (opcjonalny)
            komentarz: Komentarz do zlecenia
            
        Returns:
            ID pozycji lub None w przypadku błędu
        """
        if not self.initialized:
            logger.error("❌ MT5 nie jest zainicjalizowany")
            return None
            
        # Sprawdzenie czy handel jest dozwolony
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logger.error("❌ Błąd podczas sprawdzania statusu terminala")
            return None
            
        if not terminal_info.trade_allowed:
            logger.error("❌ Handel jest wyłączony")
            return None
            
        # Sprawdzenie typu zlecenia
        if typ not in ["BUY", "SELL"]:
            logger.error(f"❌ Nieprawidłowy typ zlecenia: {typ}")
            return None
            
        # Sprawdzenie wolumenu
        if wolumen <= 0:
            logger.error("❌ Wolumen musi być większy od 0")
            return None
            
        # Sprawdzenie czy symbol jest dostępny
        if symbol not in self.pobierz_dostepne_symbole():
            logger.error(f"❌ Symbol {symbol} nie jest dostępny")
            return None
            
        # Pobranie informacji o symbolu
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"❌ Nie znaleziono symbolu: {symbol}")
            return None
            
        try:
            # Sprawdzenie ceny
            cena = symbol_info.ask if typ == "BUY" else symbol_info.bid
            if not isinstance(cena, (int, float)) or cena <= 0:
                logger.error("❌ Nieprawidłowa cena")
                return None
                
            # Sprawdzenie dostępnych środków
            account = mt5.account_info()
            if account is None or account.balance < (cena * wolumen):
                logger.error("❌ Brak wystarczających środków")
                return None
                
            # Sprawdzenie poziomów SL/TP
            if sl is not None and tp is not None:
                if typ == "BUY":
                    if sl >= cena or tp <= cena:
                        logger.error("❌ Nieprawidłowe poziomy SL/TP dla pozycji BUY")
                        return None
                else:  # SELL
                    if sl <= cena or tp >= cena:
                        logger.error("❌ Nieprawidłowe poziomy SL/TP dla pozycji SELL")
                        return None
                    
            # Przygotowanie zlecenia
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": wolumen,
                "type": mt5.ORDER_TYPE_BUY if typ == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": cena,
                "deviation": 20,
                "magic": 234000,
                "comment": komentarz,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Dodanie SL/TP jeśli podane
            if sl:
                request["sl"] = sl
            if tp:
                request["tp"] = tp
            
            # Wysłanie zlecenia
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Błąd podczas otwierania pozycji: {result.comment}")
                return None
                
            logger.info(f"🥷 Otwarto pozycję {typ} na {symbol}, wolumen: {wolumen}")
            return result.order
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas otwierania pozycji: {str(e)}")
            return None
    
    def zamknij_pozycje(self, ticket: int) -> bool:
        """
        Zamyka istniejącą pozycję.
        
        Args:
            ticket: ID pozycji do zamknięcia
            
        Returns:
            bool: True jeśli zamknięcie się powiodło, False w przeciwnym razie
        """
        if not self.initialized:
            self.logger.error("❌ MT5 nie jest zainicjalizowany")
            return False
            
        try:
            # Pobranie informacji o pozycji
            position = mt5.positions_get(ticket=ticket)
            if not position:
                self.logger.error(f"❌ Nie znaleziono pozycji: {ticket}")
                return False
            
            position = position[0]
            
            # Przygotowanie zlecenia zamykającego
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Zamknięcie pozycji",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Wysłanie zlecenia
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"❌ Błąd podczas zamykania pozycji: {result.comment}")
                return False
                
            self.logger.info(f"🥷 Zamknięto pozycję {ticket}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas zamykania pozycji: {str(e)}")
            return False
    
    def pobierz_kalendarz(self, start_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Pobiera dane z kalendarza ekonomicznego."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
            
        try:
            events = mt5.calendar_get(start_date)
            if events is None:
                self.logger.error("❌ Nie udało się pobrać kalendarza")
                return None
                
            df = pd.DataFrame(list(events))
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas pobierania kalendarza: {e}")
            return None
    
    def zakoncz(self) -> None:
        """Zamyka połączenie z MT5."""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            logger.info("🥷 Zakończono połączenie z MT5")
    
    def _zapisz_do_bazy(self, dane: Dict[str, Any], typ: str) -> None:
        """
        Zapisuje dane do bazy jeśli jest dostępna.
        
        Args:
            dane: Dane do zapisania
            typ: Typ danych ('wiadomosc', 'sesja', 'status', 'aktyw', 'metryki', 'cena')
        """
        if not self.baza:
            return
            
        try:
            if typ == 'wiadomosc':
                self.baza.dodaj_wiadomosc(dane)
            elif typ == 'sesja':
                self.baza.dodaj_sesje_handlowa(dane)
            elif typ == 'status':
                self.baza.aktualizuj_status_rynku(dane['symbol'], dane)
            elif typ == 'aktyw':
                self.baza.aktualizuj_aktyw(dane['nazwa'], dane)
            elif typ == 'metryki':
                self.baza.dodaj_metryki(dane)
            elif typ == 'cena':
                self.baza.dodaj_cene(dane)
        except Exception as e:
            logger.error(f"❌ Błąd zapisu do bazy ({typ}): {str(e)}")
    
    def pobierz_wiadomosci(self, start_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Pobiera wiadomości z MT5."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
            
        try:
            news = mt5.news_get(start_date)
            if news is None:
                self.logger.error("❌ Nie udało się pobrać wiadomości")
                return None
                
            df = pd.DataFrame(list(news))
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas pobierania wiadomości: {e}")
            return None
    
    def pobierz_historie_konta(self, start_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Pobiera historię transakcji z konta."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
            
        try:
            deals = mt5.history_deals_get(start_date)
            if deals is None:
                self.logger.error("❌ Nie udało się pobrać historii konta")
                return None
                
            df = pd.DataFrame(list(deals))
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas pobierania historii: {e}")
            return None
    
    def pobierz_aktywa(self) -> Optional[pd.DataFrame]:
        """
        Pobiera informacje o wszystkich dostępnych aktywach.
        
        Returns:
            DataFrame z informacjami o aktywach lub None w przypadku błędu
        """
        if not self.initialized:
            logger.error("❌ MT5 nie jest zainicjalizowany")
            return None
            
        try:
            # Pobranie informacji o wszystkich symbolach
            symbols_info = []
            for symbol in self.symbols:
                info = mt5.symbol_info(symbol.name)._asdict() if mt5.symbol_info(symbol.name) else None
                if info:
                    symbols_info.append(info)
                    
                    # Zapis do bazy
                    if self.baza:
                        self._zapisz_do_bazy({
                            'nazwa': info['name'],
                            'opis': info.get('description', ''),
                            'waluta_bazowa': info.get('currency_base', ''),
                            'waluta_kwotowana': info.get('currency_profit', ''),
                            'digits': info.get('digits', 0),
                            'point': info.get('point', 0.0),
                            'tick_size': info.get('trade_tick_size', 0.0),
                            'tick_value': info.get('trade_tick_value', 0.0),
                            'lot_min': info.get('volume_min', 0.0),
                            'lot_max': info.get('volume_max', 0.0),
                            'lot_step': info.get('volume_step', 0.0),
                            'spread': info.get('spread', 0),
                            'spread_float': info.get('spread_float', False),
                            'trade_mode': info.get('trade_mode', 0),
                            'trade_stops_level': info.get('trade_stops_level', 0)
                        }, 'aktyw')
            
            # Konwersja na DataFrame
            df = pd.DataFrame(symbols_info)
            if len(df) > 0:
                logger.info(f"�� Pobrano informacje o {len(df)} aktywach")
                
                # Dodanie kolumny z statusem rynku
                df['status_rynku'] = df['name'].apply(lambda x: self.sprawdz_status_rynku(x))
                
            else:
                logger.warning("⚠️ Nie znaleziono informacji o aktywach")
                
            return df
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas pobierania informacji o aktywach: {str(e)}")
            return None
    
    def pobierz_swieta(self, symbol: str) -> Optional[pd.DataFrame]:
        """Pobiera listę świąt dla danego symbolu."""
        try:
            info = mt5.symbol_info_get(symbol)
            if info is None:
                self.logger.error(f"❌ Nie udało się pobrać informacji o symbolu {symbol}")
                return None
                
            if not hasattr(info, 'session_holidays'):
                self.logger.warning(f"⚠️ Brak informacji o świętach dla {symbol}")
                return None
                
            holidays = pd.DataFrame(info.session_holidays)
            return holidays
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas pobierania świąt: {e}")
            return None 