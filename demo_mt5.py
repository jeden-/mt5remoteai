"""
Skrypt demonstracyjny pokazujący użycie modułu operacje_mt5.py
"""

from datetime import datetime, timedelta
import pandas as pd
from handel.operacje_mt5 import OperacjeHandloweMT5
from baza_danych.baza import BazaDanych
import logging
import os
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Konfiguracja loggera
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    """Główna funkcja demonstracyjna."""
    
    # Konfiguracja bazy danych
    config_db = {
        'DB_HOST': os.getenv('DB_HOST', 'localhost'),
        'DB_PORT': os.getenv('DB_PORT', '5432'),
        'DB_NAME': os.getenv('DB_NAME', 'nikkeininja'),
        'DB_USER': os.getenv('DB_USER', 'ninja'),
        'DB_PASSWORD': os.getenv('DB_PASSWORD', 'ninja')
    }
    
    # Inicjalizacja bazy danych
    baza = BazaDanych(config_db)
    if not baza.inicjalizuj():
        logger.error("❌ Nie udało się połączyć z bazą danych")
        return
    logger.info("🥷 Połączono z bazą danych")
    
    # Inicjalizacja obiektu do operacji handlowych z bazą danych
    operacje = OperacjeHandloweMT5(baza=baza)
    if not operacje.inicjalizuj():
        logger.error("❌ Nie udało się zainicjalizować MT5")
        return
    logger.info("🥷 Połączono z MT5")
        
    try:
        # 1. Sprawdzenie dostępnych symboli
        symbole = operacje.pobierz_dostepne_symbole()
        logger.info(f"🥷 Dostępne symbole: {', '.join(symbole[:5])}...")
        
        # 2. Pobranie informacji o aktywach (automatycznie zapisze do bazy)
        df_aktywa = operacje.pobierz_aktywa()
        if df_aktywa is not None:
            logger.info("\n📊 Top 5 aktywów z najniższym spreadem:")
            top_spread = df_aktywa.nsmallest(5, 'spread')[['name', 'spread', 'trade_mode']]
            for _, row in top_spread.iterrows():
                logger.info(f"   - {row['name']}: spread={row['spread']}, tryb={row['trade_mode']}")
        
        # 3. Sprawdzenie wiadomości z ostatnich 24h (automatycznie zapisze do bazy)
        start_date = datetime.now() - timedelta(days=1)
        df_news = operacje.pobierz_wiadomosci(start_date)
        if df_news is not None and len(df_news) > 0:
            logger.info("\n📰 Ostatnie wiadomości:")
            for _, row in df_news.iterrows():
                logger.info(f"   - [{row['time'].strftime('%H:%M')}] {row['subject']} ({row['category']})")
        
        # 4. Sprawdzenie sesji handlowych dla głównych par
        glowne_pary = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD"]
        logger.info("\n🕒 Sesje handlowe dla głównych par:")
        
        for symbol in glowne_pary:
            df_sesje = operacje.pobierz_swieta(symbol)
            if df_sesje is not None and len(df_sesje) > 0:
                logger.info(f"\n   {symbol}:")
                for _, row in df_sesje.iterrows():
                    logger.info(f"   - {row['from'].strftime('%H:%M')} - {row['to'].strftime('%H:%M')}")
        
        # 5. Historia konta z ostatnich 7 dni
        start_date = datetime.now() - timedelta(days=7)
        df_historia = operacje.pobierz_historie_konta(start_date)
        if df_historia is not None and len(df_historia) > 0:
            logger.info("\n💰 Statystyki transakcji z ostatnich 7 dni:")
            zysk_calkowity = df_historia['profit'].sum()
            liczba_transakcji = len(df_historia)
            zyskowne = len(df_historia[df_historia['profit'] > 0])
            stratne = len(df_historia[df_historia['profit'] < 0])
            
            logger.info(f"   - Liczba transakcji: {liczba_transakcji}")
            logger.info(f"   - Zyskowne/Stratne: {zyskowne}/{stratne}")
            logger.info(f"   - Całkowity zysk: {zysk_calkowity:.2f}")
            
            if liczba_transakcji > 0:
                win_rate = (zyskowne / liczba_transakcji) * 100
                logger.info(f"   - Win Rate: {win_rate:.1f}%")
                
                # Zapisanie metryk do bazy
                operacje._zapisz_do_bazy({
                    'symbol': 'ALL',  # Wszystkie symbole
                    'okres_od': start_date,
                    'okres_do': datetime.now(),
                    'liczba_transakcji': liczba_transakcji,
                    'zyskowne_transakcje': zyskowne,
                    'stratne_transakcje': stratne,
                    'zysk_calkowity': zysk_calkowity,
                    'win_rate': win_rate
                }, 'metryki')
        
        # 6. Pobranie danych historycznych dla głównych par (automatycznie zapisze do bazy)
        for symbol in glowne_pary:
            logger.info(f"\n📈 Pobieranie danych historycznych dla {symbol}...")
            df_hist = operacje.pobierz_dane_historyczne(
                symbol=symbol,
                timeframe="H1",
                start_date=start_date
            )
            if df_hist is not None:
                logger.info(f"   - Pobrano {len(df_hist)} świec")
                logger.info(f"   - Zakres: {df_hist['time'].min()} - {df_hist['time'].max()}")
        
    except Exception as e:
        logger.error(f"❌ Wystąpił błąd: {str(e)}")
        
    finally:
        # Zamknięcie połączenia
        operacje.zakoncz()
        logger.info("\n👋 Zakończono demonstrację")

if __name__ == "__main__":
    main() 