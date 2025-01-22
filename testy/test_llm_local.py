"""
Testy dla modułu analizy tekstu przez API Anthropic.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from ai.llm_local import AnalizatorLLM, WynikAnalizyLLM
from ai.scraper_social import WzmiankaSocial, ZrodloSocial
from anthropic import Anthropic

@pytest.fixture
def mock_response():
    """Fixture tworzący zamockowaną odpowiedź z API."""
    response = MagicMock()
    response.content = [MagicMock(text="Sentyment: POZYTYWNY\nSłowa kluczowe: wzrost, zysk\nSugerowana akcja: KUPUJ")]
    return response

@pytest.fixture
def mock_anthropic(mock_response):
    """Fixture tworzący zamockowany klient Anthropic."""
    mock = MagicMock()
    mock.messages = MagicMock()
    mock.messages.create = AsyncMock(return_value=mock_response)
    return mock

@pytest.fixture
def analizator(mock_anthropic):
    """Fixture tworzący instancję analizatora z zamockowanym klientem."""
    original_init = Anthropic.__init__
    def mock_init(self, *args, **kwargs):
        self.messages = mock_anthropic.messages
    Anthropic.__init__ = mock_init
    try:
        return AnalizatorLLM()
    finally:
        Anthropic.__init__ = original_init

@pytest.fixture
def przykladowe_wzmianki():
    """Fixture tworzący przykładowe wzmianki do testów."""
    return [
        WzmiankaSocial(
            zrodlo=ZrodloSocial.TWITTER,
            tekst="Nikkei 225 osiąga nowe szczyty! Świetne wyniki spółek technologicznych.",
            data=datetime(2024, 1, 1),
            liczba_polubien=100,
            liczba_obserwujacych=1000
        ),
        WzmiankaSocial(
            zrodlo=ZrodloSocial.TWITTER,
            tekst="Spadki na giełdzie w Tokio. Nikkei traci przez słabe dane makro.",
            data=datetime(2024, 1, 2),
            liczba_polubien=50,
            liczba_obserwujacych=500
        )
    ]

@pytest.mark.asyncio
async def test_analiza_pozytywnego_tekstu(analizator):
    """Test analizy pozytywnego tekstu."""
    wynik = await analizator.analizuj_tekst("Nikkei 225 osiąga nowe szczyty!")
    assert wynik.sentyment == "POZYTYWNY"
    assert "wzrost" in wynik.slowa_kluczowe
    assert wynik.sugerowana_akcja == "KUPUJ"

@pytest.mark.asyncio
async def test_analiza_negatywnego_tekstu(analizator):
    """Test analizy negatywnego tekstu."""
    with patch.object(analizator.client.messages, 'create') as mock_create:
        mock_create.return_value = MagicMock(
            content=[MagicMock(text="Sentyment: NEGATYWNY\nSłowa kluczowe: spadek, strata\nSugerowana akcja: SPRZEDAJ")]
        )
        wynik = await analizator.analizuj_tekst("Nikkei 225 notuje duże spadki.")
        assert wynik.sentyment == "NEGATYWNY"
        assert "spadek" in wynik.slowa_kluczowe
        assert wynik.sugerowana_akcja == "SPRZEDAJ"

@pytest.mark.asyncio
async def test_analiza_neutralnego_tekstu(analizator):
    """Test analizy neutralnego tekstu."""
    with patch.object(analizator.client.messages, 'create') as mock_create:
        mock_create.return_value = MagicMock(
            content=[MagicMock(text="Sentyment: NEUTRALNY\nSłowa kluczowe: stabilizacja, rynek\nSugerowana akcja: CZEKAJ")]
        )
        wynik = await analizator.analizuj_tekst("Nikkei 225 pozostaje stabilny.")
        assert wynik.sentyment == "NEUTRALNY"
        assert "stabilizacja" in wynik.slowa_kluczowe
        assert wynik.sugerowana_akcja == "CZEKAJ"

@pytest.mark.asyncio
async def test_analiza_wielu_wzmianek(analizator, przykladowe_wzmianki):
    """Test analizy wielu wzmianek."""
    wyniki = await analizator.analizuj_wzmianki(przykladowe_wzmianki)
    assert len(wyniki) == 2
    assert all(isinstance(wynik, WynikAnalizyLLM) for wynik in wyniki)

@pytest.mark.asyncio
async def test_obsluga_bledow(analizator):
    """Test obsługi błędów podczas analizy."""
    with patch.object(analizator.client.messages, 'create') as mock_create:
        mock_create.side_effect = Exception("Test error")
        wynik = await analizator.analizuj_tekst("Test error")
        assert wynik is None

@pytest.mark.asyncio
async def test_parsowanie_analizy(analizator):
    """Test parsowania odpowiedzi z analizy."""
    wynik = await analizator.analizuj_tekst("Test")
    assert wynik.sentyment == "POZYTYWNY"
    assert "wzrost" in wynik.slowa_kluczowe
    assert wynik.sugerowana_akcja == "KUPUJ" 