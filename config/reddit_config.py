"""
Konfiguracja Reddit API dla systemu NikkeiNinja.
"""
import os
from dotenv import load_dotenv

# Załaduj zmienne środowiskowe
load_dotenv()

REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'), 
    'user_agent': os.getenv('REDDIT_USER_AGENT')
}

# Lista śledzonych subredditów
SUBREDDITS = [
    'wallstreetbets',
    'stocks',
    'investing',
    'trading',
    'StockMarket',
    'japan',
    'JapanFinance',
    'japanstocks'
]

# Słowa kluczowe do śledzenia
KEYWORDS = [
    'Nikkei',
    'Japan stocks',
    'JPY',
    'Japanese market',
    'Tokyo exchange',
    'N225',
    'NKY'
] 