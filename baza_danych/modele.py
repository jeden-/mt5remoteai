"""
Moduł zawierający modele (tabele) bazy danych.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, Float, String, DateTime, Enum, ForeignKey, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()

class KierunekTransakcji(enum.Enum):
    """Kierunek transakcji (long/short)."""
    LONG = "LONG"
    SHORT = "SHORT"

class StatusPozycji(enum.Enum):
    """Status pozycji tradingowej."""
    OTWARTA = "OTWARTA"
    ZAMKNIETA = "ZAMKNIETA"
    ANULOWANA = "ANULOWANA"

class Transakcja(Base):
    """Model transakcji tradingowej."""
    
    __tablename__ = "transakcje"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    kierunek = Column(Enum(KierunekTransakcji), nullable=False)
    cena = Column(Float, nullable=False)
    wolumen = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    pozycja_id = Column(Integer, ForeignKey('pozycje.id'))
    
    pozycja = relationship("Pozycja", back_populates="transakcje")

class Pozycja(Base):
    """Model pozycji tradingowej."""
    
    __tablename__ = "pozycje"
    
    id = Column(Integer, primary_key=True)
    timestamp_otwarcia = Column(DateTime, nullable=False, default=datetime.utcnow)
    timestamp_zamkniecia = Column(DateTime)
    symbol = Column(String, nullable=False)
    kierunek = Column(Enum(KierunekTransakcji), nullable=False)
    status = Column(Enum(StatusPozycji), nullable=False, default=StatusPozycji.OTWARTA)
    zysk = Column(Float)
    
    transakcje = relationship("Transakcja", back_populates="pozycja")

class HistoriaCen(Base):
    """Model historii cen."""
    
    __tablename__ = "historia_cen"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

class Wiadomosc(Base):
    """Model wiadomości z MT5."""
    
    __tablename__ = "wiadomosci"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    tytul = Column(String, nullable=False)
    kategoria = Column(String)
    typ = Column(Integer)
    priorytet = Column(Integer)
    tresc = Column(String)

class SesjaHandlowa(Base):
    """Model sesji handlowych dla symboli."""
    
    __tablename__ = "sesje_handlowe"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    od = Column(DateTime, nullable=False)
    do = Column(DateTime, nullable=False)
    typ = Column(Integer, nullable=False)  # 1=zwykła sesja, 2=święto, itp.
    opis = Column(String)

class StatusRynku(Base):
    """Model statusu rynku dla symboli."""
    
    __tablename__ = "status_rynku"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    otwarty = Column(Boolean, nullable=False)
    powod = Column(String)
    nastepne_otwarcie = Column(DateTime)

class Aktyw(Base):
    """Model informacji o aktywach."""
    
    __tablename__ = "aktywa"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    nazwa = Column(String, nullable=False)
    opis = Column(String)
    waluta_bazowa = Column(String)
    waluta_kwotowana = Column(String)
    digits = Column(Integer)  # Liczba miejsc po przecinku
    point = Column(Float)  # Wartość jednego punktu
    tick_size = Column(Float)  # Minimalny ruch ceny
    tick_value = Column(Float)  # Wartość minimalnego ruchu
    lot_min = Column(Float)  # Minimalny wolumen
    lot_max = Column(Float)  # Maksymalny wolumen
    lot_step = Column(Float)  # Krok wolumenu
    spread = Column(Float)  # Aktualny spread
    spread_float = Column(Boolean)  # Czy spread jest zmienny
    trade_mode = Column(Integer)  # Tryb handlu (0=wyłączony, 1=tylko long, 2=tylko short, 3=pełny)
    trade_stops_level = Column(Integer)  # Minimalny dystans dla SL/TP w punktach

class MetrykiHandlu(Base):
    """Model metryk handlowych dla analizy wydajności."""
    
    __tablename__ = "metryki_handlu"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    okres_od = Column(DateTime, nullable=False)
    okres_do = Column(DateTime, nullable=False)
    liczba_transakcji = Column(Integer, default=0)
    zyskowne_transakcje = Column(Integer, default=0)
    stratne_transakcje = Column(Integer, default=0)
    zysk_calkowity = Column(Float, default=0.0)
    strata_calkowita = Column(Float, default=0.0)
    najwiekszy_zysk = Column(Float, default=0.0)
    najwieksza_strata = Column(Float, default=0.0)
    sredni_zysk = Column(Float, default=0.0)
    srednia_strata = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)  # Procent zyskownych transakcji
    profit_factor = Column(Float, default=0.0)  # Stosunek zysków do strat
    sharp_ratio = Column(Float, default=0.0)  # Współczynnik Sharpe'a
    max_drawdown = Column(Float, default=0.0)  # Maksymalne obsunięcie kapitału
    max_drawdown_percent = Column(Float, default=0.0)  # Maksymalne obsunięcie w procentach

class StatusSynchronizacji(enum.Enum):
    """Status synchronizacji danych."""
    W_TOKU = "W_TOKU"
    ZAKONCZONA = "ZAKONCZONA"
    BLAD = "BLAD"

class SynchronizacjaDanych(Base):
    """Model śledzący synchronizację danych z MT5."""
    
    __tablename__ = "synchronizacja_danych"
    
    id = Column(Integer, primary_key=True)
    timestamp_start = Column(DateTime, nullable=False, default=datetime.utcnow)
    timestamp_koniec = Column(DateTime)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    zakres_od = Column(DateTime, nullable=False)
    zakres_do = Column(DateTime, nullable=False)
    status = Column(Enum(StatusSynchronizacji), nullable=False, default=StatusSynchronizacji.W_TOKU)
    liczba_rekordow = Column(Integer, default=0)
    blad = Column(String)
    
    __table_args__ = (
        Index('idx_sync_symbol_tf', 'symbol', 'timeframe'),
    )

class Cache(Base):
    """Model cache'u dla często używanych danych."""
    
    __tablename__ = "cache"
    
    id = Column(Integer, primary_key=True)
    klucz = Column(String, nullable=False, unique=True)
    wartosc = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    wygasa = Column(DateTime)
    
    __table_args__ = (
        Index('idx_cache_klucz', 'klucz'),
        Index('idx_cache_wygasa', 'wygasa'),
    )

class ZadanieAktualizacji(Base):
    """Model zadań aktualizacji danych."""
    
    __tablename__ = "zadania_aktualizacji"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    typ = Column(String, nullable=False)  # np. 'historia_cen', 'sesje', 'status'
    priorytet = Column(Integer, default=1)
    wykonane = Column(Boolean, default=False)
    blad = Column(String)
    
    __table_args__ = (
        Index('idx_zadania_status', 'wykonane', 'priorytet'),
    )

class KalendarzEkonomiczny(Base):
    """Model dla wydarzeń z kalendarza ekonomicznego."""
    
    __tablename__ = 'kalendarz_ekonomiczny'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    event_id = Column(Integer, nullable=False)
    nazwa = Column(String, nullable=False)
    waluta = Column(String, nullable=False)
    kraj = Column(String, nullable=False)
    waznosc = Column(Integer, nullable=False)
    wartosc_aktualna = Column(Float)
    wartosc_prognoza = Column(Float)
    wartosc_poprzednia = Column(Float)
    rewizja = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_kalendarz_timestamp', 'timestamp'),
        Index('idx_kalendarz_event', 'event_id'),
        Index('idx_kalendarz_waluta', 'waluta'),
        Index('idx_kalendarz_kraj', 'kraj'),
        Index('idx_kalendarz_waznosc', 'waznosc')
    ) 