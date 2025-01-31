"""
Moduł zawierający testy dla promptów.
"""
import pytest
from src.utils.prompts import (
    get_ollama_prompt_template,
    get_claude_prompt_template,
    format_market_data_for_prompt,
    TradingPrompts
)


def test_get_ollama_prompt_template():
    """Test szablonu promptu dla Ollama."""
    prompt = get_ollama_prompt_template()
    
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert '{symbol}' in prompt
    assert '{current_price}' in prompt
    assert '{sma_20}' in prompt
    assert '{sma_50}' in prompt


def test_get_claude_prompt_template():
    """Test szablonu promptu dla Claude."""
    prompt = get_claude_prompt_template()
    
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert '{symbol}' in prompt
    assert '{current_price}' in prompt
    assert '{sma_20}' in prompt
    assert '{sma_50}' in prompt


def test_format_market_data_for_prompt():
    """Test formatowania danych rynkowych do promptu."""
    market_data = {
        'symbol': 'EURUSD',
        'current_price': 1.1,
        'sma_20': 1.09,
        'sma_50': 1.08,
        'price_change_24h': 0.01,
        'volume_24h': 1000.0
    }
    
    formatted_data = format_market_data_for_prompt(market_data)
    
    assert isinstance(formatted_data, str)
    assert 'EURUSD' in formatted_data
    assert '1.1' in formatted_data
    assert '1.09' in formatted_data
    assert '1.08' in formatted_data
    assert '0.01' in formatted_data
    assert '1000.0' in formatted_data


def test_format_market_data_with_missing_values():
    """Test formatowania danych z brakującymi wartościami."""
    market_data = {
        'symbol': 'EURUSD',
        'current_price': 1.1
    }
    
    formatted_data = format_market_data_for_prompt(market_data)
    
    assert isinstance(formatted_data, str)
    assert 'EURUSD' in formatted_data
    assert '1.1' in formatted_data
    assert 'N/A' in formatted_data  # Dla brakujących wartości


def test_format_market_data_with_none_values():
    """Test formatowania danych z wartościami None."""
    market_data = {
        'symbol': 'EURUSD',
        'current_price': 1.1,
        'sma_20': None,
        'sma_50': None
    }
    
    formatted_data = format_market_data_for_prompt(market_data)
    
    assert isinstance(formatted_data, str)
    assert 'EURUSD' in formatted_data
    assert '1.1' in formatted_data
    assert 'N/A' in formatted_data


def test_format_market_data_with_extra_fields():
    """Test formatowania danych z dodatkowymi polami."""
    market_data = {
        'symbol': 'EURUSD',
        'current_price': 1.1,
        'sma_20': 1.09,
        'sma_50': 1.08,
        'extra_field': 'test'  # Dodatkowe pole
    }
    
    formatted_data = format_market_data_for_prompt(market_data)
    
    assert isinstance(formatted_data, str)
    assert 'EURUSD' in formatted_data
    assert '1.1' in formatted_data
    assert 'test' not in formatted_data  # Dodatkowe pole nie powinno być uwzględnione


def test_ollama_prompt_template_format():
    """Test formatowania szablonu promptu Ollama."""
    prompt = get_ollama_prompt_template()
    market_data = {
        'symbol': 'EURUSD',
        'current_price': 1.1,
        'sma_20': 1.09,
        'sma_50': 1.08
    }
    
    formatted_prompt = prompt.format(**market_data)
    
    assert 'EURUSD' in formatted_prompt
    assert '1.1' in formatted_prompt
    assert '1.09' in formatted_prompt
    assert '1.08' in formatted_prompt


def test_claude_prompt_template_format():
    """Test formatowania szablonu promptu Claude."""
    prompt = get_claude_prompt_template()
    market_data = {
        'symbol': 'EURUSD',
        'current_price': 1.1,
        'sma_20': 1.09,
        'sma_50': 1.08
    }
    
    formatted_prompt = prompt.format(**market_data)
    
    assert 'EURUSD' in formatted_prompt
    assert '1.1' in formatted_prompt
    assert '1.09' in formatted_prompt
    assert '1.08' in formatted_prompt


def test_prompt_templates_differences():
    """Test różnic między szablonami promptów."""
    ollama_prompt = get_ollama_prompt_template()
    claude_prompt = get_claude_prompt_template()
    
    assert ollama_prompt != claude_prompt  # Szablony powinny się różnić


def test_format_market_data_validation():
    """Test walidacji danych wejściowych."""
    with pytest.raises(ValueError):
        format_market_data_for_prompt(None)
        
    with pytest.raises(ValueError):
        format_market_data_for_prompt({})  # Pusty słownik
        
    with pytest.raises(ValueError):
        format_market_data_for_prompt({'symbol': None})


def test_trading_prompts_market_analysis():
    """Test generowania promptu do analizy rynku."""
    data = {
        'symbol': 'EURUSD',
        'current_price': 1.1234,
        'sma_20': 1.1200,
        'sma_50': 1.1180,
        'price_change_24h': 0.12,
        'volume_24h': 1000000
    }
    
    prompt = TradingPrompts.get_market_analysis_prompt(data)
    
    assert isinstance(prompt, str)
    assert 'EURUSD' in prompt
    assert '1.1234' in prompt
    assert '1.1200' in prompt
    assert '1.1180' in prompt
    assert '0.12' in prompt
    assert '1000000' in prompt
    assert 'TREND_DIRECTION' in prompt
    assert 'TREND_STRENGTH' in prompt
    assert 'RECOMMENDATION' in prompt


def test_trading_prompts_market_analysis_validation():
    """Test walidacji danych dla analizy rynku."""
    with pytest.raises(ValueError):
        TradingPrompts.get_market_analysis_prompt({})
        
    with pytest.raises(ValueError):
        TradingPrompts.get_market_analysis_prompt({'symbol': 'EURUSD'})  # Brak wymaganych pól


def test_trading_prompts_risk_analysis():
    """Test generowania promptu do analizy ryzyka."""
    data = {
        'symbol': 'EURUSD',
        'type': 'BUY',
        'volume': 0.1,
        'entry_price': 1.1234,
        'account_balance': 10000,
        'current_exposure': 0.05
    }
    
    prompt = TradingPrompts.get_risk_analysis_prompt(data)
    
    assert isinstance(prompt, str)
    assert 'EURUSD' in prompt
    assert 'BUY' in prompt
    assert '0.10' in prompt
    assert '1.1234' in prompt
    assert '10000.00' in prompt
    assert '5.00%' in prompt
    assert 'RISK_LEVEL' in prompt
    assert 'RR_RATIO' in prompt
    assert 'CAPITAL_AT_RISK' in prompt


def test_trading_prompts_risk_analysis_validation():
    """Test walidacji danych dla analizy ryzyka."""
    with pytest.raises(ValueError):
        TradingPrompts.get_risk_analysis_prompt({})
        
    with pytest.raises(ValueError):
        TradingPrompts.get_risk_analysis_prompt({'symbol': 'EURUSD'})  # Brak wymaganych pól 