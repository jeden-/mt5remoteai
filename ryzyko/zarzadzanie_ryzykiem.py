"""
Moduł odpowiedzialny za zarządzanie ryzykiem.
"""

import logging
from typing import Dict, List, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)

class ZarzadzanieRyzykiem:
    """Klasa implementująca zarządzanie ryzykiem."""
    
    def __init__(self, max_ryzyko_trade: float = 0.02, max_ryzyko_dzienne: float = 0.06):
        """
        Inicjalizacja zarządzania ryzykiem.
        
        Args:
            max_ryzyko_trade: Maksymalne ryzyko na transakcję (2%)
            max_ryzyko_dzienne: Maksymalne ryzyko dzienne (6%)
        """
        self.logger = logger
        self.max_ryzyko_trade = Decimal(str(max_ryzyko_trade))
        self.max_ryzyko_dzienne = Decimal(str(max_ryzyko_dzienne))
        self.dzienne_ryzyko = Decimal('0')
        
    def oblicz_ryzyko(self, wielkosc_pozycji: float, stop_loss: float, cena: float) -> Dict[str, Any]:
        """
        Oblicza ryzyko dla danej pozycji.
        
        Args:
            wielkosc_pozycji: Wielkość pozycji
            stop_loss: Poziom stop loss
            cena: Aktualna cena
            
        Returns:
            Dict[str, Any]: Informacje o ryzyku
        """
        ryzyko = abs(cena - stop_loss) * wielkosc_pozycji
        procent_ryzyko = Decimal(str(ryzyko)) / Decimal(str(cena))
        
        return {
            "ryzyko_kwotowe": float(ryzyko),
            "ryzyko_procentowe": float(procent_ryzyko),
            "dozwolone": procent_ryzyko <= self.max_ryzyko_trade
        }
        
    def aktualizuj_dzienne_ryzyko(self, ryzyko: float) -> Dict[str, Any]:
        """
        Aktualizuje dzienne ryzyko.
        
        Args:
            ryzyko: Ryzyko do dodania
            
        Returns:
            Dict[str, Any]: Status aktualizacji
        """
        nowe_ryzyko = self.dzienne_ryzyko + Decimal(str(ryzyko))
        dozwolone = nowe_ryzyko <= self.max_ryzyko_dzienne
        
        if dozwolone:
            self.dzienne_ryzyko = nowe_ryzyko
            
        return {
            "dozwolone": dozwolone,
            "aktualne_ryzyko": float(self.dzienne_ryzyko),
            "pozostale_ryzyko": float(self.max_ryzyko_dzienne - self.dzienne_ryzyko)
        } 