"""
Interfejs dla strategii tradingowych w systemie NikkeiNinja.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class KierunekTransakcji(Enum):
    """Kierunek transakcji (długa/krótka)."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class SygnalTransakcyjny:
    """Sygnał wygenerowany przez strategię."""
    timestamp: datetime
    symbol: str
    kierunek: KierunekTransakcji
    cena_wejscia: float
    stop_loss: float
    take_profit: float
    wolumen: float
    opis: str
    metadane: Dict = None


class IStrategia(ABC):
    """Interfejs dla strategii tradingowych."""
    
    @abstractmethod
    def inicjalizuj(self, parametry: Dict) -> None:
        """
        Inicjalizuje strategię z podanymi parametrami.
        
        Args:
            parametry: Słownik z parametrami strategii
        """
        pass
    
    @abstractmethod
    def analizuj(self, 
                 df: pd.DataFrame,
                 dodatkowe_dane: Optional[Dict] = None) -> List[SygnalTransakcyjny]:
        """
        Analizuje dane i generuje sygnały transakcyjne.
        
        Args:
            df: DataFrame z danymi rynkowymi (OHLCV)
            dodatkowe_dane: Opcjonalny słownik z dodatkowymi danymi (np. sentyment)
            
        Returns:
            Lista sygnałów transakcyjnych
        """
        pass
    
    @abstractmethod
    def aktualizuj(self, 
                   nowe_dane: pd.DataFrame,
                   aktywne_pozycje: List[Tuple[str, KierunekTransakcji, float]]) -> List[SygnalTransakcyjny]:
        """
        Aktualizuje strategię o nowe dane i sprawdza aktywne pozycje.
        
        Args:
            nowe_dane: DataFrame z nowymi danymi rynkowymi
            aktywne_pozycje: Lista krotek (symbol, kierunek, cena_wejscia)
            
        Returns:
            Lista sygnałów do zamknięcia/modyfikacji pozycji
        """
        pass
    
    @abstractmethod
    def optymalizuj(self, 
                    dane_historyczne: pd.DataFrame,
                    parametry_zakres: Dict[str, Tuple[float, float, float]]) -> Dict:
        """
        Optymalizuje parametry strategii na danych historycznych.
        
        Args:
            dane_historyczne: DataFrame z danymi do optymalizacji
            parametry_zakres: Słownik z zakresami parametrów (min, max, krok)
            
        Returns:
            Słownik z optymalnymi parametrami
        """
        pass
    
    @abstractmethod
    def generuj_statystyki(self, 
                          historia_transakcji: List[SygnalTransakcyjny]) -> Dict:
        """
        Generuje statystyki skuteczności strategii.
        
        Args:
            historia_transakcji: Lista historycznych sygnałów
            
        Returns:
            Słownik ze statystykami (win rate, profit factor, etc.)
        """
        pass 