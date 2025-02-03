"""
Moduł zawierający testy dla konektora Anthropic.
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
from anthropic import AuthenticationError
from src.connectors.anthropic_connector import AnthropicConnector


@pytest.fixture
def config():
    """Fixture tworzący przykładową konfigurację."""
    return {
        "api": {
            "anthropic_key": "test_key"
        }
    }


@pytest.fixture
def mock_anthropic():
    """Fixture tworzący mock dla klienta Anthropic."""
    mock_response = Mock()
    mock_response.content = [Mock(text="Market analysis result")]
    
    mock_messages = MagicMock()
    mock_messages.create.return_value = mock_response
    
    mock_client = MagicMock()
    mock_client.messages = mock_messages
    
    with patch('anthropic.Anthropic', return_value=mock_client) as mock_anthropic_class:
        mock_anthropic_class.return_value = mock_client
        yield mock_client


def test_initialization(config):
    """Test inicjalizacji konektora."""
    connector = AnthropicConnector(config)
    assert connector.client is not None


def test_initialization_missing_api_key():
    """Test inicjalizacji bez klucza API."""
    with pytest.raises(ValueError) as exc_info:
        AnthropicConnector({})
    assert "Brak klucza API Anthropic w konfiguracji" in str(exc_info.value)


@pytest.mark.asyncio
async def test_analyze_market_conditions_success(mock_anthropic, config):
    """Test udanej analizy warunków rynkowych."""
    # Przygotuj dane testowe
    connector = AnthropicConnector(config)
    connector.client = mock_anthropic  # Podmień klienta na mocka
    
    market_data = {
        'symbol': 'EURUSD',
        'price': 1.1234,
        'sma_20': 1.12
    }
    prompt_template = "Analyze {symbol} at price {price} with SMA20 at {sma_20}"
    
    # Wywołaj metodę
    result = await connector.analyze_market_conditions(market_data, prompt_template)
    
    # Sprawdź wyniki
    assert result == "Market analysis result"
    mock_anthropic.messages.create.assert_called_once_with(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[{
            'role': 'user',
            'content': "Analyze EURUSD at price 1.1234 with SMA20 at 1.12"
        }]
    )


@pytest.mark.asyncio
async def test_analyze_market_conditions_auth_error(mock_anthropic, config):
    """Test błędu autoryzacji."""
    connector = AnthropicConnector(config)
    connector.client = mock_anthropic  # Podmień klienta na mocka
    
    market_data = {'symbol': 'EURUSD'}
    prompt_template = "Analyze {symbol}"
    
    # Skonfiguruj mock aby rzucał wyjątek autoryzacji
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.json.return_value = {
        'type': 'error',
        'error': {
            'type': 'authentication_error',
            'message': 'invalid x-api-key'
        }
    }
    mock_anthropic.messages.create.side_effect = AuthenticationError(
        message="Error code: 401 - invalid x-api-key",
        response=mock_response,
        body=mock_response.json()
    )
    
    # Sprawdź czy metoda rzuca wyjątek
    with pytest.raises(AuthenticationError) as exc_info:
        await connector.analyze_market_conditions(market_data, prompt_template)
    assert "invalid x-api-key" in str(exc_info.value)


@pytest.mark.asyncio
async def test_analyze_market_conditions_general_error(mock_anthropic, config):
    """Test ogólnego błędu API."""
    connector = AnthropicConnector(config)
    connector.client = mock_anthropic  # Podmień klienta na mocka
    
    market_data = {'symbol': 'EURUSD'}
    prompt_template = "Analyze {symbol}"
    
    # Skonfiguruj mock aby rzucał ogólny wyjątek
    mock_anthropic.messages.create.side_effect = Exception("Internal server error")
    
    # Sprawdź czy metoda rzuca wyjątek
    with pytest.raises(Exception) as exc_info:
        await connector.analyze_market_conditions(market_data, prompt_template)
    assert "Internal server error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_analyze_market_conditions_invalid_template(mock_anthropic, config):
    """Test nieprawidłowego szablonu promptu."""
    connector = AnthropicConnector(config)
    connector.client = mock_anthropic  # Podmień klienta na mocka
    
    market_data = {'symbol': 'EURUSD'}
    prompt_template = "Analyze {invalid_field}"  # Pole nie istnieje w market_data
    
    # Sprawdź czy metoda rzuca wyjątek
    with pytest.raises(KeyError):
        await connector.analyze_market_conditions(market_data, prompt_template) 