"""
Testy dla modułu analizy sentymentu.
"""

import pytest
from datetime import datetime
from ai.analiza_sentymentu import AnalizatorSentymentu, WynikSentymentu
from ai.scraper_social import WzmiankaSocial, ZrodloSocial

@pytest.fixture
def analizator():
    """Fixture dostarczający instancję analizatora sentymentu."""
    return AnalizatorSentymentu()

@pytest.fixture
def przykladowe_wzmianki():
    """Fixture dostarczający przykładowe wzmianki do testów."""
    return [
        WzmiankaSocial(
            tekst="Nikkei 225 showing strong bullish momentum!",
            data=datetime.now(),
            zrodlo=ZrodloSocial.TWITTER,
            liczba_polubien=100,
            liczba_obserwujacych=1000
        ),
        WzmiankaSocial(
            tekst="Nikkei 225 in a downward trend",
            data=datetime.now(),
            zrodlo=ZrodloSocial.TWITTER,
            liczba_polubien=50,
            liczba_obserwujacych=500
        ),
        WzmiankaSocial(
            tekst="Nikkei 225 trading sideways",
            data=datetime.now(),
            zrodlo=ZrodloSocial.TWITTER,
            liczba_polubien=75,
            liczba_obserwujacych=750
        )
    ]

def test_inicjalizacja_analizatora(analizator):
    """Test poprawnej inicjalizacji analizatora."""
    assert analizator is not None
    assert analizator.model is not None
    assert analizator.tokenizer is not None

def test_mapowanie_etykiet(analizator):
    """Test mapowania etykiet na wartości numeryczne."""
    assert analizator.mapuj_etykiete("positive") == 1.0
    assert analizator.mapuj_etykiete("negative") == -1.0
    assert analizator.mapuj_etykiete("neutral") == 0.0
    # Test dla etykiet z dużych liter
    assert analizator.mapuj_etykiete("POSITIVE") == 1.0
    assert analizator.mapuj_etykiete("NEGATIVE") == -1.0

@pytest.mark.asyncio
async def test_analiza_pozytywnego_tekstu(analizator):
    """Test analizy pozytywnego tekstu."""
    tekst = "Nikkei 225 showing strong bullish momentum, great buying opportunity!"
    wynik = await analizator.analizuj_tekst(tekst)
    assert wynik is not None
    assert wynik.etykieta in ["positive", "negative", "neutral"]
    assert 0.0 <= wynik.pewnosc <= 1.0
    assert -1.0 <= wynik.wartosc_numeryczna <= 1.0

@pytest.mark.asyncio
async def test_analiza_negatywnego_tekstu(analizator):
    """Test analizy negatywnego tekstu."""
    tekst = "Nikkei 225 in trouble, bearish signals everywhere"
    wynik = await analizator.analizuj_tekst(tekst)
    assert wynik is not None
    assert wynik.etykieta in ["positive", "negative", "neutral"]
    assert 0.0 <= wynik.pewnosc <= 1.0
    assert -1.0 <= wynik.wartosc_numeryczna <= 1.0

@pytest.mark.asyncio
async def test_analiza_neutralnego_tekstu(analizator):
    """Test analizy neutralnego tekstu."""
    tekst = "Nikkei 225 trading sideways in a narrow range"
    wynik = await analizator.analizuj_tekst(tekst)
    assert wynik is not None
    assert wynik.etykieta in ["positive", "negative", "neutral"]
    assert 0.0 <= wynik.pewnosc <= 1.0
    assert -1.0 <= wynik.wartosc_numeryczna <= 1.0

@pytest.mark.asyncio
async def test_analiza_wielu_wzmianek(analizator, przykladowe_wzmianki):
    """Test analizy wielu wzmianek."""
    wyniki = await analizator.analizuj_wzmianki(przykladowe_wzmianki)
    assert isinstance(wyniki, dict)
    assert "sredni_sentyment" in wyniki
    assert "pozytywne_proc" in wyniki
    assert "neutralne_proc" in wyniki
    assert "negatywne_proc" in wyniki
    assert -1.0 <= wyniki["sredni_sentyment"] <= 1.0
    assert 0.0 <= wyniki["pozytywne_proc"] <= 100.0
    assert 0.0 <= wyniki["neutralne_proc"] <= 100.0
    assert 0.0 <= wyniki["negatywne_proc"] <= 100.0
    assert abs(wyniki["pozytywne_proc"] + wyniki["neutralne_proc"] + wyniki["negatywne_proc"] - 100.0) < 0.01

@pytest.mark.asyncio
async def test_analiza_pustej_listy(analizator):
    """Test analizy pustej listy wzmianek."""
    wyniki = await analizator.analizuj_wzmianki([])
    assert isinstance(wyniki, dict)
    assert wyniki["sredni_sentyment"] == 0.0
    assert wyniki["pozytywne_proc"] == 0.0
    assert wyniki["neutralne_proc"] == 0.0
    assert wyniki["negatywne_proc"] == 0.0

@pytest.mark.asyncio
async def test_obsluga_bledow(analizator):
    """Test obsługi błędów podczas analizy."""
    wynik = await analizator.analizuj_tekst(None)
    assert wynik is None 