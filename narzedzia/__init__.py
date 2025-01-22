"""
Pakiet zawierający narzędzia pomocnicze systemu.
"""

from .logowanie import skonfiguruj_logger
from .konfiguracja import wczytaj_config

__all__ = ['skonfiguruj_logger', 'wczytaj_config'] 