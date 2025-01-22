"""
Moduł odpowiedzialny za analizę sentymentu wzmianek z mediów społecznościowych.
Wykorzystuje transformers do oceny wydźwięku tekstu.
"""

import logging
from typing import List, Dict, Optional
from transformers import pipeline
from dataclasses import dataclass
from .scraper_social import WzmiankaSocial

# Konfiguracja loggera
logger = logging.getLogger(__name__)

class WynikSentymentu:
    """Klasa reprezentująca wynik analizy sentymentu."""
    def __init__(self, etykieta: str, pewnosc: float):
        self.etykieta = etykieta.lower()
        self.pewnosc = pewnosc
        self.wartosc_numeryczna = self.mapuj_etykiete(self.etykieta)
    
    @staticmethod
    def mapuj_etykiete(etykieta: str) -> float:
        """Mapuje etykietę tekstową na wartość numeryczną."""
        if etykieta == "positive":
            return 1.0
        elif etykieta == "negative":
            return -1.0
        return 0.0

class AnalizatorSentymentu:
    """Klasa do analizy sentymentu tekstu."""
    
    def __init__(self):
        """Inicjalizuje analizator sentymentu."""
        self.model = pipeline("sentiment-analysis")
        self.tokenizer = self.model.tokenizer
        self.logger = logging.getLogger(__name__)
    
    def mapuj_etykiete(self, etykieta: str) -> float:
        """Mapuje etykietę na wartość numeryczną."""
        return WynikSentymentu.mapuj_etykiete(etykieta.lower())
    
    async def analizuj_tekst(self, tekst: str) -> Optional[WynikSentymentu]:
        """
        Analizuje sentyment podanego tekstu.
        
        Args:
            tekst: Tekst do analizy
            
        Returns:
            Obiekt WynikSentymentu lub None w przypadku błędu
        """
        try:
            wynik = self.model(tekst)[0]
            return WynikSentymentu(wynik["label"], wynik["score"])
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas analizy tekstu: {str(e)}")
            return None
    
    async def analizuj_wzmianki(self, wzmianki: List[WzmiankaSocial]) -> Dict[str, float]:
        """
        Analizuje sentyment listy wzmianek i oblicza zagregowane statystyki.
        
        Args:
            wzmianki: Lista obiektów WzmiankaSocial do analizy
            
        Returns:
            Słownik ze statystykami sentymentu
        """
        statystyki = {
            "sredni_sentyment": 0.0,
            "pozytywne_proc": 0.0,
            "neutralne_proc": 0.0,
            "negatywne_proc": 0.0
        }
        
        if not wzmianki:
            return statystyki
            
        licznik = {"positive": 0, "neutral": 0, "negative": 0}
        suma_sentymentu = 0.0
        przeanalizowane = 0
        
        for wzmianka in wzmianki:
            wynik = await self.analizuj_tekst(wzmianka.tekst)
            if wynik:
                licznik[wynik.etykieta] += 1
                suma_sentymentu += wynik.wartosc_numeryczna
                przeanalizowane += 1
        
        if przeanalizowane > 0:
            statystyki["sredni_sentyment"] = suma_sentymentu / przeanalizowane
            statystyki["pozytywne_proc"] = (licznik["positive"] / przeanalizowane) * 100
            statystyki["neutralne_proc"] = (licznik["neutral"] / przeanalizowane) * 100
            statystyki["negatywne_proc"] = (licznik["negative"] / przeanalizowane) * 100
            
        self.logger.info(
            f"🥷 Przeanalizowano {przeanalizowane} wzmianek. "
            f"Średni sentyment: {statystyki['sredni_sentyment']:.2f}"
        )
            
        return statystyki 