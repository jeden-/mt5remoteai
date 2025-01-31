"""
Moduł zawierający klasę do komunikacji z Anthropic API.
"""
from anthropic import Anthropic, AuthenticationError
import httpx
from typing import Dict, Any
import asyncio
from loguru import logger

from ..utils.config import Config


class AnthropicConnector:
    """Klasa odpowiedzialna za komunikację z API Anthropic."""

    def __init__(self, config: Config):
        """
        Inicjalizacja konektora Anthropic.

        Args:
            config: Obiekt konfiguracyjny z kluczem API
        """
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        
    async def analyze_market_conditions(self, market_data: Dict[str, Any], prompt_template: str) -> str:
        """
        Analizuje warunki rynkowe używając Claude'a.
        
        Args:
            market_data: Słownik zawierający dane rynkowe
            prompt_template: Szablon promptu do analizy
            
        Returns:
            str: Odpowiedź Claude'a
            
        Raises:
            AuthenticationError: Gdy wystąpi błąd uwierzytelniania
            Exception: Gdy wystąpi inny błąd w komunikacji z API
        """
        try:
            prompt = prompt_template.format(**market_data)
            
            logger.debug(f"🥷 Wysyłanie zapytania do Claude: {prompt[:100]}...")
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            result = response.content[0].text
            logger.debug(f"🥷 Otrzymano odpowiedź od Claude: {result[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas analizy danych przez Claude: {str(e)}")
            raise 