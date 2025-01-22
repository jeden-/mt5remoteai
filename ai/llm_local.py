"""Moduł do analizy tekstu przez API Anthropic.

Moduł wykorzystuje model Claude do analizy sentymentu, ekstrakcji kluczowych informacji
i sugerowania akcji tradingowych na podstawie wzmianek w mediach społecznościowych.
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from anthropic import Anthropic
from .scraper_social import WzmiankaSocial

# Ładowanie zmiennych środowiskowych
load_dotenv()

@dataclass
class WynikAnalizyLLM:
    """Struktura przechowująca wynik analizy tekstu."""
    sentyment: str
    slowa_kluczowe: List[str]
    sugerowana_akcja: str

class AnalizatorLLM:
    """Klasa do analizy tekstu przy użyciu modelu Claude."""

    def __init__(self, api_key: Optional[str] = None):
        """Inicjalizacja analizatora z kluczem API."""
        self.logger = logging.getLogger(__name__)
        self.client = Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            _strict_response_validation=False
        )
        self.model = "claude-3-opus-20240229"

    async def analizuj_tekst(self, tekst: str) -> Optional[WynikAnalizyLLM]:
        """Analizuje tekst używając modelu Claude i zwraca szczegółowy wynik analizy."""
        try:
            prompt = f"""Przeanalizuj poniższy tekst dotyczący indeksu Nikkei 225 i podaj:
1. Sentyment (POZYTYWNY/NEGATYWNY/NEUTRALNY)
2. 2-3 kluczowe słowa
3. Sugerowaną akcję (KUPUJ/SPRZEDAJ/CZEKAJ)

Tekst: {tekst}

Odpowiedź sformatuj następująco:
Sentyment: [SENTYMENT]
Słowa kluczowe: [SLOWO1, SLOWO2]
Sugerowana akcja: [AKCJA]"""

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Pobieranie tekstu z odpowiedzi
            response_text = response.content[0].text

            # Parsowanie odpowiedzi
            lines = response_text.strip().split('\n')
            sentyment = lines[0].split(': ')[1]
            slowa_kluczowe = [s.strip() for s in lines[1].split(': ')[1].split(',')]
            sugerowana_akcja = lines[2].split(': ')[1]

            return WynikAnalizyLLM(
                sentyment=sentyment,
                slowa_kluczowe=slowa_kluczowe,
                sugerowana_akcja=sugerowana_akcja
            )
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas analizy tekstu: {str(e)}")
            return None

    async def analizuj_wzmianki(self, wzmianki: List[WzmiankaSocial]) -> List[WynikAnalizyLLM]:
        """Analizuje listę wzmianek i zwraca wyniki dla każdej z nich."""
        wyniki = []
        for wzmianka in wzmianki:
            try:
                wynik = await self.analizuj_tekst(wzmianka.tekst)
                if wynik:
                    wyniki.append(wynik)
            except Exception as e:
                self.logger.error(f"❌ Błąd podczas analizy wzmianki: {str(e)}")
        return wyniki 