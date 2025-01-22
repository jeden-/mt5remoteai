"""
Moduł odpowiedzialny za pobieranie i analizę danych z mediów społecznościowych.
Wykorzystuje snscrape do monitorowania wzmianek o Nikkei225.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import snscrape.modules.twitter as sntwitter
from dataclasses import dataclass
from enum import Enum, auto

# Konfiguracja loggera
logger = logging.getLogger(__name__)

class ZrodloSocial(Enum):
    """Dostępne źródła danych społecznościowych."""
    TWITTER = auto()

@dataclass
class WzmiankaSocial:
    """Struktura przechowująca informacje o wzmiance w mediach społecznościowych."""
    tekst: str
    data: datetime
    zrodlo: ZrodloSocial
    liczba_polubien: int
    liczba_obserwujacych: int
    autor: Optional[str] = None

class ScraperSocial:
    """Klasa do pobierania wzmianek z mediów społecznościowych."""

    def __init__(self):
        """Inicjalizacja scrapera."""
        self.hashtagi = ["#Nikkei225", "#NikkeiIndex", "#JapanStocks"]
        self.slowa_kluczowe = ["Nikkei 225", "Japanese market"]
        self.min_likes = 5
        self.min_followers = 100

    async def pobierz_wzmianki_twitter(self, limit: int = 100) -> List[WzmiankaSocial]:
        """Pobiera wzmianki z Twittera."""
        wzmianki = []
        query = " OR ".join(self.hashtagi + self.slowa_kluczowe)
        
        try:
            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                if len(wzmianki) >= limit:
                    break
                    
                if tweet.likeCount >= self.min_likes and tweet.user.followersCount >= self.min_followers:
                    wzmianka = WzmiankaSocial(
                        zrodlo=ZrodloSocial.TWITTER,
                        tekst=tweet.rawContent,
                        data=tweet.date,
                        autor=tweet.user.username,
                        liczba_polubien=tweet.likeCount,
                        liczba_obserwujacych=tweet.user.followersCount
                    )
                    wzmianki.append(wzmianka)
                    logger.info(f"🥷 Pobrano wzmiankę z Twittera od {wzmianka.autor}")
        except Exception as e:
            logger.error(f"❌ Błąd podczas pobierania wzmianek z Twittera: {str(e)}")
        
        return wzmianki

    async def aktualizuj_dane(self) -> List[WzmiankaSocial]:
        """Pobiera najnowsze wzmianki ze wszystkich źródeł."""
        wzmianki = []
        
        # Pobierz wzmianki z Twittera
        wzmianki_twitter = await self.pobierz_wzmianki_twitter()
        wzmianki.extend(wzmianki_twitter)
        
        logger.info(f"🥷 Zaktualizowano dane społecznościowe - pobrano {len(wzmianki)} wzmianek")
        return wzmianki 