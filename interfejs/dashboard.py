"""
Moduł implementujący dashboard systemu.
"""

import logging
from typing import Dict, Any
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class DashboardApp:
    """Klasa implementująca dashboard systemu."""
    
    def __init__(self, port: int = 8050):
        """
        Inicjalizacja dashboardu.
        
        Args:
            port: Port na którym uruchomić aplikację
        """
        self.logger = logger
        self.port = port
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY]
        )
        self._skonfiguruj_layout()
        
    def _skonfiguruj_layout(self):
        """Konfiguruje układ dashboardu."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("NikkeiNinja Dashboard 🥷", className="text-center mb-4"))
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="wykres-ceny"),
                    dcc.Interval(
                        id="interval-component",
                        interval=1000,  # ms
                        n_intervals=0
                    )
                ])
            ])
        ], fluid=True)
        
    def uruchom(self):
        """Uruchamia aplikację dashboard."""
        self.app.run_server(debug=True, port=self.port) 