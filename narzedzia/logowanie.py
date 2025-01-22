"""
Moduł konfiguracji logowania.
"""

import logging
import logging.handlers
import os
from typing import Optional
import yaml

def skonfiguruj_logger(
    nazwa: str,
    poziom: str = "INFO",
    plik_logow: Optional[str] = None
) -> logging.Logger:
    """
    Konfiguruje i zwraca logger.
    
    Args:
        nazwa: Nazwa loggera
        poziom: Poziom logowania
        plik_logow: Ścieżka do pliku logów
        
    Returns:
        logging.Logger: Skonfigurowany logger
    """
    # Tworzenie loggera
    logger = logging.getLogger(nazwa)
    logger.setLevel(getattr(logging, poziom.upper()))
    
    # Format logów
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler konsolowy
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler plikowy (opcjonalny)
    if plik_logow:
        os.makedirs(os.path.dirname(plik_logow), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            plik_logow,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 