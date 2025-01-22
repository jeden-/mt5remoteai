"""
Testy dla modułu rdzen.
"""

import pytest
from rdzen.polaczenie_mt5 import PolaczenieMT5

def test_inicjalizacja_polaczenia():
    """Test inicjalizacji połączenia z MT5."""
    connector = PolaczenieMT5()
    assert connector is not None
    assert connector.polaczony is False
    assert connector.handel_dozwolony is False
    
@pytest.mark.asyncio
async def test_polaczenie_mt5():
    """Test połączenia z MT5."""
    connector = PolaczenieMT5()
    wynik = connector.inicjalizuj()
    
    assert wynik["status"] is True
    assert wynik["wersja"] is not None
    assert isinstance(wynik["polaczony"], bool)
    assert isinstance(wynik["handel_dozwolony"], bool)
    assert wynik["sciezka_terminala"] is not None
    
    connector.zakoncz() 