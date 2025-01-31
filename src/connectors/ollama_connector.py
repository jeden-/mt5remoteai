"""
Moduł zawierający klasę do komunikacji z Ollama API.
"""
import requests
from typing import Dict, Any
from loguru import logger


class OllamaConnector:
    """Klasa odpowiedzialna za komunikację z lokalnym modelem Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Inicjalizacja konektora Ollama.

        Args:
            base_url: Bazowy URL do API Ollama (domyślnie http://localhost:11434)
        """
        self.base_url = base_url
        
    async def analyze_market_data(self, market_data: Dict[str, Any], prompt_template: str) -> str:
        """
        Analizuje dane rynkowe używając lokalnego modelu Ollama.
        
        Args:
            market_data: Słownik zawierający dane rynkowe
            prompt_template: Szablon promptu do analizy
            
        Returns:
            str: Odpowiedź modelu
            
        Raises:
            Exception: Gdy wystąpi błąd w komunikacji z API
        """
        try:
            prompt = prompt_template.format(**market_data)
            
            logger.debug(f"🥷 Wysyłanie zapytania do Ollama: {prompt[:100]}...")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "mistral",  # lub inny model dostępny w Ollama
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()['response']
                logger.debug(f"🥷 Otrzymano odpowiedź od Ollama: {result[:100]}...")
                return result
            else:
                error_msg = f"Błąd Ollama API: {response.status_code}"
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"❌ Błąd podczas analizy danych przez Ollama: {str(e)}")
            raise 