"""
Skrypt demonstracyjny pokazujący użycie RedditScraper.
"""
import logging
from datetime import datetime
from scrapery.reddit_scraper import RedditScraper

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def wyswietl_post(post):
    """Wyświetla szczegóły pojedynczego posta."""
    print(f"\n📝 Post z r/{post['subreddit']}")
    print(f"Tytuł: {post['tytul']}")
    print(f"Autor: u/{post['autor']}")
    print(f"Data: {post['utworzony'].strftime('%Y-%m-%d %H:%M')}")
    print(f"Score: {post['score']} | Komentarze: {post['komentarze']} | Ratio: {post['upvote_ratio']:.2%}")
    if post['tresc']:
        print(f"Treść: {post['tresc'][:200]}...")

def main():
    """Główna funkcja demonstracyjna."""
    scraper = RedditScraper()
    
    # 1. Najnowsze posty z r/japanstocks
    print("\n🗾 === Najnowsze posty z r/japanstocks ===")
    posty_japan = scraper.pobierz_posty('japanstocks', limit=3)
    for post in posty_japan:
        wyswietl_post(post)
        
    # 2. Wyszukiwanie wzmianek o Nikkei
    print("\n📈 === Wyszukiwanie 'Nikkei' na wszystkich subredditach ===")
    posty_nikkei = scraper.szukaj_posty('Nikkei', limit=3)
    for post in posty_nikkei:
        wyswietl_post(post)
    
    # 3. Analiza sentymentu dla postów o japońskim rynku
    print("\n📊 === Analiza sentymentu dla postów o 'Japan stocks' ===")
    posty_japan_stocks = scraper.szukaj_posty('Japan stocks', limit=10)
    statystyki = scraper.analizuj_sentyment(posty_japan_stocks)
    
    if statystyki:
        print(f"\nStatystyki dla {statystyki['liczba_postow']} postów:")
        print(f"📈 Średni score: {statystyki['sredni_score']:.2f}")
        print(f"💬 Średnia liczba komentarzy: {statystyki['srednia_liczba_komentarzy']:.2f}")
        print(f"👍 Średni upvote ratio: {statystyki['sredni_upvote_ratio']:.2%}")
        print(f"🕒 Zakres dat: {statystyki['najstarszy_post'].strftime('%Y-%m-%d')} - {statystyki['najnowszy_post'].strftime('%Y-%m-%d')}")
    
    # 4. Wyszukiwanie wzmianek o JPY
    print("\n💴 === Wyszukiwanie wzmianek o 'JPY' ===")
    posty_jpy = scraper.szukaj_posty('JPY', limit=3)
    for post in posty_jpy:
        wyswietl_post(post)

if __name__ == '__main__':
    main() 