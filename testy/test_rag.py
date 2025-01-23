"""
Test systemu RAG (Retrieval Augmented Generation).
"""

import pytest
from datetime import datetime
from ai.system_rag import SystemRAG

@pytest.fixture
def system_rag():
    """Fixture dostarczający instancję SystemRAG."""
    return SystemRAG()

def test_dodawanie_dokumentu(system_rag):
    """Test dodawania dokumentu do bazy wiedzy."""
    dokument = {
        'tekst': 'Nikkei 225 osiągnął nowe maksima po pozytywnych danych makro z Japonii.',
        'metadata': {
            'data': datetime.now().isoformat(),
            'zrodlo': 'test',
            'typ': 'wiadomosc'
        }
    }
    
    assert system_rag.dodaj_dokument(dokument['tekst'], dokument['metadata']) == True

def test_wyszukiwanie(system_rag):
    """Test wyszukiwania dokumentów."""
    # Dodanie kilku dokumentów testowych
    dokumenty = [
        {
            'tekst': 'Nikkei 225 osiągnął nowe maksima po pozytywnych danych makro z Japonii.',
            'metadata': {'data': '2024-01-23', 'zrodlo': 'test1', 'typ': 'wiadomosc'}
        },
        {
            'tekst': 'Bank Japonii utrzymał stopy procentowe bez zmian.',
            'metadata': {'data': '2024-01-23', 'zrodlo': 'test2', 'typ': 'wiadomosc'}
        },
        {
            'tekst': 'Spółki technologiczne z indeksu Nikkei notują wzrosty.',
            'metadata': {'data': '2024-01-23', 'zrodlo': 'test3', 'typ': 'wiadomosc'}
        }
    ]
    
    for dok in dokumenty:
        system_rag.dodaj_dokument(dok['tekst'], dok['metadata'])
    
    # Test wyszukiwania
    wyniki = system_rag.wyszukaj("Nikkei wzrosty")
    assert len(wyniki) > 0
    assert any("Nikkei" in wynik['tekst'] for wynik in wyniki)

def test_wyszukiwanie_z_filtrem(system_rag):
    """Test wyszukiwania z filtrowaniem po metadanych."""
    # Dodanie dokumentów z różnymi typami
    dokumenty = [
        {
            'tekst': 'Analiza techniczna Nikkei 225 wskazuje na trend wzrostowy.',
            'metadata': {'data': '2024-01-23', 'zrodlo': 'test1', 'typ': 'analiza'}
        },
        {
            'tekst': 'Bank Japonii utrzymał stopy procentowe bez zmian.',
            'metadata': {'data': '2024-01-23', 'zrodlo': 'test2', 'typ': 'wiadomosc'}
        }
    ]
    
    for dok in dokumenty:
        system_rag.dodaj_dokument(dok['tekst'], dok['metadata'])
    
    # Test wyszukiwania tylko analiz
    wyniki = system_rag.wyszukaj(
        "trend Nikkei",
        filtry={'typ': 'analiza'}
    )
    assert len(wyniki) > 0
    assert all(wynik['metadata']['typ'] == 'analiza' for wynik in wyniki)

def test_aktualizacja_dokumentu(system_rag):
    """Test aktualizacji istniejącego dokumentu."""
    dokument = {
        'tekst': 'Stara wersja dokumentu.',
        'metadata': {
            'id': 'test123',
            'data': '2024-01-23',
            'wersja': 1
        }
    }
    
    # Dodanie dokumentu
    system_rag.dodaj_dokument(dokument['tekst'], dokument['metadata'])
    
    # Aktualizacja dokumentu
    dokument_v2 = {
        'tekst': 'Nowa wersja dokumentu.',
        'metadata': {
            'id': 'test123',
            'data': '2024-01-23',
            'wersja': 2
        }
    }
    
    assert system_rag.dodaj_dokument(
        dokument_v2['tekst'],
        dokument_v2['metadata'],
        aktualizuj=True
    ) == True
    
    # Sprawdzenie czy mamy tylko nową wersję
    wyniki = system_rag.wyszukaj("dokument")
    assert len([w for w in wyniki if w['metadata']['id'] == 'test123']) == 1
    assert any(w['metadata']['wersja'] == 2 for w in wyniki if w['metadata']['id'] == 'test123') 