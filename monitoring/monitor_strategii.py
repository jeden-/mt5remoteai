"""
Moduł do monitorowania statystyk strategii w czasie rzeczywistym.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque

from baza_danych.baza import BazaDanych
from strategie.interfejs import SygnalTransakcyjny, KierunekTransakcji

logger = logging.getLogger(__name__)

class MonitorStrategii:
    """
    Klasa do monitorowania statystyk strategii w czasie rzeczywistym.
    Przechowuje historię transakcji i generuje statystyki na żywo.
    """
    
    def __init__(self, baza: BazaDanych, okno_czasu: timedelta = timedelta(hours=24)):
        """
        Inicjalizacja monitora strategii.
        
        Args:
            baza: Instancja bazy danych
            okno_czasu: Okno czasowe dla statystyk (domyślnie 24h)
        """
        self.baza = baza
        self.okno_czasu = okno_czasu
        self.historia = deque(maxlen=1000)  # Ostatnie 1000 transakcji
        self.statystyki_cache = {}
        self.ostatnia_aktualizacja = datetime.now()
        logger.info("🥷 Zainicjalizowano monitor strategii")
    
    def dodaj_sygnal(self, sygnal: SygnalTransakcyjny) -> None:
        """
        Dodaje nowy sygnał do historii i aktualizuje statystyki.
        
        Args:
            sygnal: Nowy sygnał transakcyjny
        """
        try:
            self.historia.append(sygnal)
            self._aktualizuj_statystyki()
            
            # Zapisz metryki do bazy
            self.baza.dodaj_metryki({
                'timestamp': sygnal.timestamp,
                'symbol': sygnal.symbol,
                'kierunek': sygnal.kierunek.value,
                'zysk_procent': sygnal.metadane.get('zysk_procent', 0),
                'sentyment': sygnal.metadane.get('sentyment', 0),
                'sentyment_pewnosc': sygnal.metadane.get('sentyment_pewnosc', 0),
                'waga': sygnal.metadane.get('waga', 0)
            })
            
            logger.info("🥷 Dodano nowy sygnał do monitora")
            
        except Exception as e:
            logger.error("❌ Błąd podczas dodawania sygnału: %s", str(e))
    
    def _aktualizuj_statystyki(self) -> None:
        """Aktualizuje statystyki na podstawie historii transakcji."""
        try:
            teraz = datetime.now()
            historia_w_oknie = [
                s for s in self.historia 
                if s.timestamp >= teraz - self.okno_czasu
            ]
            
            if not historia_w_oknie:
                return
            
            # Podstawowe statystyki
            liczba_transakcji = len(historia_w_oknie)
            zyskowne = sum(1 for s in historia_w_oknie 
                          if s.metadane.get('zysk_procent', 0) > 0)
            
            # Statystyki per kierunek
            long_transakcje = [s for s in historia_w_oknie 
                             if s.kierunek == KierunekTransakcji.LONG]
            short_transakcje = [s for s in historia_w_oknie 
                              if s.kierunek == KierunekTransakcji.SHORT]
            
            # Statystyki sentymentu
            sredni_sentyment = np.mean([
                s.metadane.get('sentyment', 0) 
                for s in historia_w_oknie
            ])
            
            self.statystyki_cache = {
                'timestamp': teraz,
                'okno_czasu_h': self.okno_czasu.total_seconds() / 3600,
                'liczba_transakcji': liczba_transakcji,
                'win_rate': zyskowne / liczba_transakcji if liczba_transakcji > 0 else 0,
                'liczba_long': len(long_transakcje),
                'liczba_short': len(short_transakcje),
                'win_rate_long': (sum(1 for s in long_transakcje 
                                    if s.metadane.get('zysk_procent', 0) > 0) 
                                 / len(long_transakcje) if long_transakcje else 0),
                'win_rate_short': (sum(1 for s in short_transakcje 
                                     if s.metadane.get('zysk_procent', 0) > 0) 
                                  / len(short_transakcje) if short_transakcje else 0),
                'sredni_sentyment': float(sredni_sentyment),
                'sredni_zysk': float(np.mean([
                    s.metadane.get('zysk_procent', 0) 
                    for s in historia_w_oknie
                ])),
                'max_zysk': float(max([
                    s.metadane.get('zysk_procent', 0) 
                    for s in historia_w_oknie
                ])),
                'max_strata': float(min([
                    s.metadane.get('zysk_procent', 0) 
                    for s in historia_w_oknie
                ])),
                'srednia_waga': float(np.mean([
                    s.metadane.get('waga', 0) 
                    for s in historia_w_oknie
                ]))
            }
            
            self.ostatnia_aktualizacja = teraz
            logger.debug("🥷 Zaktualizowano statystyki monitora")
            
        except Exception as e:
            logger.error("❌ Błąd podczas aktualizacji statystyk: %s", str(e))
    
    def pobierz_statystyki(self) -> Dict:
        """
        Pobiera aktualne statystyki.
        
        Returns:
            Dict ze statystykami
        """
        # Aktualizuj statystyki jeśli cache jest starszy niż 5 minut
        if (datetime.now() - self.ostatnia_aktualizacja) > timedelta(minutes=5):
            self._aktualizuj_statystyki()
        
        return self.statystyki_cache
    
    def generuj_raport(self) -> str:
        """
        Generuje tekstowy raport ze statystykami.
        
        Returns:
            str: Raport w formacie tekstowym
        """
        try:
            stats = self.pobierz_statystyki()
            if not stats:
                return "Brak danych do wygenerowania raportu"
            
            return f"""📊 Raport strategii (ostatnie {stats['okno_czasu_h']}h)
            
Podstawowe statystyki:
- Liczba transakcji: {stats['liczba_transakcji']}
- Win rate: {stats['win_rate']:.2%}
- Średni zysk: {stats['sredni_zysk']:.2f}%
- Max zysk: {stats['max_zysk']:.2f}%
- Max strata: {stats['max_strata']:.2f}%

Kierunki:
- Long: {stats['liczba_long']} (WR: {stats['win_rate_long']:.2%})
- Short: {stats['liczba_short']} (WR: {stats['win_rate_short']:.2%})

Sentyment:
- Średni sentyment: {stats['sredni_sentyment']:.2f}
- Średnia waga sygnałów: {stats['srednia_waga']:.2f}
"""
            
        except Exception as e:
            logger.error("❌ Błąd podczas generowania raportu: %s", str(e))
            return "Błąd podczas generowania raportu" 