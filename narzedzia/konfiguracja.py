"""
Moduł obsługi konfiguracji.
"""

import os
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

def wczytaj_config(sciezka: str = "config/ustawienia.yaml") -> Dict[str, Any]:
    """
    Wczytuje konfigurację z pliku YAML i zmiennych środowiskowych.
    
    Args:
        sciezka: Ścieżka do pliku konfiguracyjnego
        
    Returns:
        Dict[str, Any]: Słownik z konfiguracją
    """
    # Wczytanie zmiennych środowiskowych
    load_dotenv()
    
    # Wczytanie konfiguracji z pliku YAML
    with open(sciezka, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Podmiana zmiennych środowiskowych
    def _podmien_env(wartosc):
        if isinstance(wartosc, str) and wartosc.startswith("${") and wartosc.endswith("}"):
            zmienna = wartosc[2:-1]
            return os.getenv(zmienna)
        return wartosc
    
    def _przejdz_rekurencyjnie(dane):
        if isinstance(dane, dict):
            return {k: _przejdz_rekurencyjnie(v) for k, v in dane.items()}
        elif isinstance(dane, list):
            return [_przejdz_rekurencyjnie(item) for item in dane]
        else:
            return _podmien_env(dane)
    
    return _przejdz_rekurencyjnie(config) 