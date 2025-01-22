"""
Moduł odpowiedzialny za obliczanie rozmiaru pozycji.
"""

import logging
from typing import Dict, List, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)

class RozmiarPozycji:
    """Klasa implementująca obliczanie rozmiaru pozycji."""
    
    def __init__(self, kapital: float, max_ryzyko: float = 0.02):
        """
        Inicjalizacja kalkulatora rozmiaru pozycji.
        
        Args:
            kapital: Dostępny kapitał
            max_ryzyko: Maksymalne ryzyko na pozycję (2%)
        """
        self.logger = logger
        self.kapital = Decimal(str(kapital))
        self.max_ryzyko = Decimal(str(max_ryzyko))
        
    def oblicz_rozmiar(self, cena: float, stop_loss: float) -> Dict[str, Any]:
        """
        Oblicza optymalny rozmiar pozycji.
        
        Args:
            cena: Aktualna cena
            stop_loss: Poziom stop loss
            
        Returns:
            Dict[str, Any]: Informacje o rozmiarze pozycji
        """
        ryzyko_kwotowe = self.kapital * self.max_ryzyko
        dystans_sl = abs(Decimal(str(cena)) - Decimal(str(stop_loss)))
        
        if dystans_sl == 0:
            return {
                "sukces": False,
                "blad": "Dystans stop loss nie może być zerowy"
            }
            
        rozmiar = ryzyko_kwotowe / dystans_sl
        
        return {
            "sukces": True,
            "rozmiar": float(rozmiar),
            "ryzyko_kwotowe": float(ryzyko_kwotowe),
            "ryzyko_procentowe": float(self.max_ryzyko)
        } 