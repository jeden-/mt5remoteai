"""
Moduł do zbierania danych z Reddita w systemie NikkeiNinja.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import praw
from praw.models import Submission, Subreddit

logger = logging.getLogger(__name__)


class RedditScraper:
    """
    Klasa do zbierania i analizy danych z Reddita.
    Obsługuje popularne subreddity finansowe:
    - r/wallstreetbets
    - r/stocks
    - r/investing
    - r/trading
    - r/StockMarket
    """
    
    def __init__(self):
        """Inicjalizacja scrapera."""
        self.reddit = None
        self.subreddits = [
            'wallstreetbets',
            'stocks',
            'investing',
            'trading',
            'StockMarket'
        ]
        logger.info("🥷 Zainicjalizowano Reddit Scraper")
    
    def inicjalizuj(self, credentials: Dict) -> None:
        """
        Inicjalizuje połączenie z Reddit API.
        
        Parametry:
        - credentials: słownik z danymi uwierzytelniającymi:
            - client_id: ID aplikacji
            - client_secret: Secret aplikacji
            - user_agent: Nazwa agenta (np. 'NikkeiNinja/1.0')
        """
        try:
            self.reddit = praw.Reddit(
                client_id=credentials['client_id'],
                client_secret=credentials['client_secret'],
                user_agent=credentials['user_agent']
            )
            logger.info("🥷 Połączono z Reddit API")
            
        except Exception as e:
            logger.error("❌ Błąd podczas inicjalizacji Reddit API: %s", str(e))
            raise
    
    def _analizuj_post(self, post: Submission) -> Dict:
        """Analizuje pojedynczy post z Reddita."""
        return {
            'id': post.id,
            'subreddit': post.subreddit.display_name,
            'title': post.title,
            'text': post.selftext,
            'score': post.score,
            'upvote_ratio': post.upvote_ratio,
            'num_comments': post.num_comments,
            'created_utc': datetime.fromtimestamp(post.created_utc),
            'url': post.url,
            'author': str(post.author),
            'is_original_content': post.is_original_content,
            'awards': len(post.all_awardings) if hasattr(post, 'all_awardings') else 0
        }
    
    def pobierz_posty(self, 
                      subreddits: Optional[List[str]] = None,
                      limit: int = 100,
                      sort: str = 'hot',
                      time_filter: str = 'day') -> List[Dict]:
        """
        Pobiera posty z wybranych subredditów.
        
        Parametry:
        - subreddits: lista subredditów (domyślnie wszystkie zdefiniowane)
        - limit: maksymalna liczba postów do pobrania
        - sort: sposób sortowania ('hot', 'new', 'top', 'controversial')
        - time_filter: filtr czasowy dla 'top' i 'controversial' ('hour', 'day', 'week', 'month', 'year', 'all')
        """
        if not self.reddit:
            raise RuntimeError("Reddit API nie zostało zainicjalizowane")
            
        subreddits = subreddits or self.subreddits
        posty = []
        
        try:
            for subreddit_name in subreddits:
                logger.info("🥷 Pobieram posty z r/%s", subreddit_name)
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Wybór metody sortowania
                if sort == 'hot':
                    submissions = subreddit.hot(limit=limit)
                elif sort == 'new':
                    submissions = subreddit.new(limit=limit)
                elif sort == 'top':
                    submissions = subreddit.top(time_filter=time_filter, limit=limit)
                elif sort == 'controversial':
                    submissions = subreddit.controversial(time_filter=time_filter, limit=limit)
                else:
                    raise ValueError(f"Nieznana metoda sortowania: {sort}")
                
                # Analiza postów
                for post in submissions:
                    try:
                        post_data = self._analizuj_post(post)
                        posty.append(post_data)
                    except Exception as e:
                        logger.warning("⚠️ Błąd podczas analizy posta %s: %s", post.id, str(e))
                        continue
                
                logger.info("🥷 Pobrano %d postów z r/%s", len(posty), subreddit_name)
                
        except Exception as e:
            logger.error("❌ Błąd podczas pobierania postów: %s", str(e))
            
        return posty
    
    def szukaj_posty(self,
                     query: str,
                     subreddits: Optional[List[str]] = None,
                     limit: int = 100) -> List[Dict]:
        """
        Wyszukuje posty zawierające określone słowa kluczowe.
        
        Parametry:
        - query: fraza do wyszukania
        - subreddits: lista subredditów (domyślnie wszystkie zdefiniowane)
        - limit: maksymalna liczba postów do pobrania
        """
        if not self.reddit:
            raise RuntimeError("Reddit API nie zostało zainicjalizowane")
            
        subreddits = subreddits or self.subreddits
        posty = []
        
        try:
            for subreddit_name in subreddits:
                logger.info("🥷 Szukam '%s' w r/%s", query, subreddit_name)
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for post in subreddit.search(query, limit=limit):
                    try:
                        post_data = self._analizuj_post(post)
                        posty.append(post_data)
                    except Exception as e:
                        logger.warning("⚠️ Błąd podczas analizy posta %s: %s", post.id, str(e))
                        continue
                
                logger.info("🥷 Znaleziono %d postów w r/%s", len(posty), subreddit_name)
                
        except Exception as e:
            logger.error("❌ Błąd podczas wyszukiwania postów: %s", str(e))
            
        return posty
    
    def analizuj_sentyment(self, posty: List[Dict]) -> Dict:
        """
        Analizuje sentyment zebranych postów.
        
        Parametry:
        - posty: lista postów do analizy
        
        Zwraca:
        - Słownik ze statystykami sentymentu
        """
        if not posty:
            return {}
            
        try:
            # Podstawowe statystyki
            statystyki = {
                'liczba_postow': len(posty),
                'sredni_score': sum(p['score'] for p in posty) / len(posty),
                'sredni_upvote_ratio': sum(p['upvote_ratio'] for p in posty) / len(posty),
                'srednia_liczba_komentarzy': sum(p['num_comments'] for p in posty) / len(posty),
                'posty_per_subreddit': {}
            }
            
            # Statystyki per subreddit
            for post in posty:
                subreddit = post['subreddit']
                if subreddit not in statystyki['posty_per_subreddit']:
                    statystyki['posty_per_subreddit'][subreddit] = {
                        'liczba_postow': 0,
                        'suma_score': 0,
                        'suma_komentarzy': 0
                    }
                
                statystyki['posty_per_subreddit'][subreddit]['liczba_postow'] += 1
                statystyki['posty_per_subreddit'][subreddit]['suma_score'] += post['score']
                statystyki['posty_per_subreddit'][subreddit]['suma_komentarzy'] += post['num_comments']
            
            return statystyki
            
        except Exception as e:
            logger.error("❌ Błąd podczas analizy sentymentu: %s", str(e))
            return {} 