"""
Moduł zawierający testy dla konektora Ollama.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.connectors.ollama_connector import OllamaConnector


@pytest.fixture
def connector():
    """Fixture tworzący instancję konektora."""
    return OllamaConnector()


def test_initialization():
    """Test inicjalizacji konektora."""
    # Test z domyślnym URL
    connector = OllamaConnector()
    assert connector.base_url == "http://localhost:11434"
    
    # Test z niestandardowym URL
    custom_url = "http://custom:11434"
    connector = OllamaConnector(base_url=custom_url)
    assert connector.base_url == custom_url


@pytest.mark.asyncio
@patch('requests.post')
async def test_analyze_market_data_success(mock_post):
    """Test udanej analizy danych rynkowych."""
    # Przygotuj dane testowe
    connector = OllamaConnector()
    market_data = {
        'symbol': 'EURUSD',
        'price': 1.1234,
        'sma_20': 1.12
    }
    prompt_template = "Analyze {symbol} at price {price} with SMA20 at {sma_20}"
    expected_response = "Market analysis result"
    
    # Skonfiguruj mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': expected_response}
    mock_post.return_value = mock_response
    
    # Wywołaj metodę
    result = await connector.analyze_market_data(market_data, prompt_template)
    
    # Sprawdź wyniki
    assert result == expected_response
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": "Analyze EURUSD at price 1.1234 with SMA20 at 1.12",
            "stream": False
        }
    )


@pytest.mark.asyncio
@patch('requests.post')
async def test_analyze_market_data_api_error(mock_post):
    """Test błędu API podczas analizy."""
    connector = OllamaConnector()
    market_data = {'symbol': 'EURUSD'}
    prompt_template = "Analyze {symbol}"
    
    # Skonfiguruj mock aby zwracał błąd
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response
    
    # Sprawdź czy metoda rzuca wyjątek
    with pytest.raises(Exception, match="Błąd Ollama API: 500"):
        await connector.analyze_market_data(market_data, prompt_template)


@pytest.mark.asyncio
@patch('requests.post')
async def test_analyze_market_data_request_error(mock_post):
    """Test błędu żądania HTTP podczas analizy."""
    connector = OllamaConnector()
    market_data = {'symbol': 'EURUSD'}
    prompt_template = "Analyze {symbol}"
    
    # Skonfiguruj mock aby rzucał wyjątek
    mock_post.side_effect = Exception("Connection error")
    
    # Sprawdź czy metoda rzuca wyjątek
    with pytest.raises(Exception):
        await connector.analyze_market_data(market_data, prompt_template)


@pytest.mark.asyncio
@patch('requests.post')
async def test_analyze_market_data_invalid_template(mock_post):
    """Test nieprawidłowego szablonu promptu."""
    connector = OllamaConnector()
    market_data = {'symbol': 'EURUSD'}
    prompt_template = "Analyze {invalid_field}"  # Pole nie istnieje w market_data
    
    # Sprawdź czy metoda rzuca wyjątek
    with pytest.raises(KeyError):
        await connector.analyze_market_data(market_data, prompt_template)


@pytest.mark.asyncio
@patch('requests.post')
async def test_analyze_market_data_invalid_response(mock_post):
    """Test nieprawidłowej odpowiedzi API."""
    connector = OllamaConnector()
    market_data = {'symbol': 'EURUSD'}
    prompt_template = "Analyze {symbol}"
    
    # Skonfiguruj mock aby zwracał nieprawidłową odpowiedź
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}  # Brak klucza 'response'
    mock_post.return_value = mock_response
    
    # Sprawdź czy metoda rzuca wyjątek
    with pytest.raises(KeyError):
        await connector.analyze_market_data(market_data, prompt_template) 