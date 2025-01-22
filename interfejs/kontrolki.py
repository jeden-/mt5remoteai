"""
Moduł implementujący kontrolki interfejsu użytkownika.
"""

import logging
from typing import Dict, Any, Callable
import dash_bootstrap_components as dbc
from dash import html

logger = logging.getLogger(__name__)

class Kontrolki:
    """Klasa implementująca kontrolki interfejsu."""
    
    def __init__(self):
        """Inicjalizacja kontrolek."""
        self.logger = logger
        
    def przycisk_start(self, callback: Callable) -> html.Button:
        """
        Tworzy przycisk startu.
        
        Args:
            callback: Funkcja wywoływana po kliknięciu
            
        Returns:
            html.Button: Przycisk startu
        """
        return dbc.Button(
            "Start Trading 🚀",
            id="przycisk-start",
            color="success",
            className="me-1"
        )
        
    def przycisk_stop(self, callback: Callable) -> html.Button:
        """
        Tworzy przycisk stopu.
        
        Args:
            callback: Funkcja wywoływana po kliknięciu
            
        Returns:
            html.Button: Przycisk stopu
        """
        return dbc.Button(
            "Stop Trading ⛔",
            id="przycisk-stop",
            color="danger",
            className="me-1"
        )
        
    def slider_ryzyko(self, callback: Callable) -> dbc.Row:
        """
        Tworzy slider do kontroli ryzyka.
        
        Args:
            callback: Funkcja wywoływana przy zmianie
            
        Returns:
            dbc.Row: Komponent slidera
        """
        return dbc.Row([
            dbc.Label("Maksymalne ryzyko na pozycję"),
            dbc.Col([
                dbc.Input(
                    type="range",
                    min=0.01,
                    max=0.05,
                    step=0.01,
                    value=0.02,
                    id="slider-ryzyko"
                )
            ])
        ]) 