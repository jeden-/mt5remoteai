"""
Moduł zawierający modele (tabele) bazy danych.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, Float, String, DateTime, Enum, ForeignKey, Boolean
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