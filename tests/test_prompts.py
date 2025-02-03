"""
Testy jednostkowe dla modułu prompts.py
"""
import pytest
from decimal import Decimal
from src.utils.prompts import (
    get_ollama_prompt_template,
    get_claude_prompt_template,
    format_market_data_for_prompt,
    TradingPrompts
)


def test_get_ollama_prompt_template():
    """Test szablonu promptu dla modelu Ollama."""
    template = get_ollama_prompt_template()
    assert isinstance(template, str)
    assert "{symbol}" in template
    assert "{current_price}" in template
    assert "{sma_20}" in template
    assert "{sma_50}" in template
    assert "TREND:" in template
    assert "STRENGTH:" in template
    assert "RECOMMENDATION:" in template
    assert "REASONING:" in template


def test_get_claude_prompt_template():
    """Test szablonu promptu dla modelu Claude."""
    template = get_claude_prompt_template()
    assert isinstance(template, str)
    assert "{symbol}" in template
    assert "{current_price}" in template
    assert "{sma_20}" in template
    assert "{sma_50}" in template
    assert "Rekomendację" in template
    assert "Sugerowany SL" in template
    assert "Sugerowany TP" in template
    assert "Uzasadnienie" in template


def test_format_market_data_for_prompt_valid_data():
    """Test formatowania poprawnych danych rynkowych."""
    market_data = {
        'symbol': 'EURUSD',
        'current_price': 1.1234,
        'sma_20': 1.1200,
        'sma_50': 1.1180,
        'price_change_24h': 0.5,
        'volume_24h': 1000000
    }
    
    formatted = format_market_data_for_prompt(market_data)
    assert isinstance(formatted, str)
    assert "EURUSD" in formatted
    assert "1.1234" in formatted
    assert "1.12" in formatted  # SMA 20
    assert "1.118" in formatted  # SMA 50
    assert "0.5" in formatted
    assert "1000000" in formatted


def test_format_market_data_for_prompt_missing_optional():
    """Test formatowania danych z brakującymi opcjonalnymi polami."""
    market_data = {
        'symbol': 'EURUSD',
        'current_price': 1.1234
    }
    
    formatted = format_market_data_for_prompt(market_data)
    assert isinstance(formatted, str)
    assert "EURUSD" in formatted
    assert "1.1234" in formatted
    assert "N/A" in formatted  # Brakujące pola powinny być zastąpione przez N/A


def test_format_market_data_for_prompt_none_values():
    """Test formatowania danych z wartościami None."""
    market_data = {
        'symbol': 'EURUSD',
        'current_price': None,
        'sma_20': None,
        'sma_50': None,
        'price_change_24h': None,
        'volume_24h': None
    }
    
    formatted = format_market_data_for_prompt(market_data)
    assert isinstance(formatted, str)
    assert "EURUSD" in formatted
    assert formatted.count("N/A") == 5  # Wszystkie None powinny być zamienione na N/A


def test_format_market_data_for_prompt_invalid_input():
    """Test formatowania dla nieprawidłowych danych wejściowych."""
    with pytest.raises(ValueError, match="Market data must be a non-empty dictionary"):
        format_market_data_for_prompt(None)
    
    with pytest.raises(ValueError, match="Market data must be a non-empty dictionary"):
        format_market_data_for_prompt({})
    
    with pytest.raises(ValueError, match="Symbol is required in market data"):
        format_market_data_for_prompt({'current_price': 1.1234})


def test_trading_prompts_market_analysis_valid_data():
    """Test generowania promptu analizy rynku z poprawnymi danymi."""
    data = {
        'symbol': 'EURUSD',
        'current_price': 1.1234,
        'sma_20': 1.1200,
        'sma_50': 1.1180,
        'price_change_24h': 0.5,
        'volume_24h': 1000000
    }
    
    prompt = TradingPrompts.get_market_analysis_prompt(data)
    assert isinstance(prompt, str)
    assert "EURUSD" in prompt
    assert "1.1234" in prompt
    assert "1.1200" in prompt
    assert "1.1180" in prompt
    assert "0.5" in prompt
    assert "1000000" in prompt
    assert "TREND_DIRECTION:" in prompt
    assert "TREND_STRENGTH:" in prompt
    assert "RECOMMENDATION:" in prompt
    assert "SL_PIPS:" in prompt
    assert "TP_PIPS:" in prompt
    assert "REASONING:" in prompt


def test_trading_prompts_market_analysis_decimal_values():
    """Test generowania promptu analizy rynku z wartościami Decimal."""
    data = {
        'symbol': 'EURUSD',
        'current_price': Decimal('1.1234'),
        'sma_20': Decimal('1.1200'),
        'sma_50': Decimal('1.1180'),
        'price_change_24h': Decimal('0.5'),
        'volume_24h': 1000000
    }
    
    prompt = TradingPrompts.get_market_analysis_prompt(data)
    assert isinstance(prompt, str)
    assert "1.1234" in prompt
    assert "1.1200" in prompt
    assert "1.1180" in prompt
    assert "0.50" in prompt


def test_trading_prompts_market_analysis_missing_fields():
    """Test generowania promptu analizy rynku z brakującymi polami."""
    data = {
        'symbol': 'EURUSD',
        'current_price': 1.1234
    }
    
    with pytest.raises(ValueError, match="Brak wymaganego pola:"):
        TradingPrompts.get_market_analysis_prompt(data)


def test_trading_prompts_market_analysis_empty_data():
    """Test generowania promptu analizy rynku z pustymi danymi."""
    with pytest.raises(ValueError, match="Brak wymaganego pola:"):
        TradingPrompts.get_market_analysis_prompt({})


def test_trading_prompts_risk_analysis_valid_data():
    """Test generowania promptu analizy ryzyka z poprawnymi danymi."""
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
    assert "EURUSD" in prompt
    assert "BUY" in prompt
    assert "0.10" in prompt
    assert "1.1234" in prompt
    assert "10000.00" in prompt
    assert "5.00%" in prompt
    assert "RISK_LEVEL:" in prompt
    assert "RR_RATIO:" in prompt
    assert "CAPITAL_AT_RISK:" in prompt
    assert "RECOMMENDATION:" in prompt
    assert "REASONING:" in prompt


def test_trading_prompts_risk_analysis_decimal_values():
    """Test generowania promptu analizy ryzyka z wartościami Decimal."""
    data = {
        'symbol': 'EURUSD',
        'type': 'SELL',
        'volume': Decimal('0.1'),
        'entry_price': Decimal('1.1234'),
        'account_balance': Decimal('10000'),
        'current_exposure': Decimal('0.05')
    }
    
    prompt = TradingPrompts.get_risk_analysis_prompt(data)
    assert isinstance(prompt, str)
    assert "0.10" in prompt
    assert "1.1234" in prompt
    assert "10000.00" in prompt
    assert "5.00%" in prompt


def test_trading_prompts_risk_analysis_missing_fields():
    """Test generowania promptu analizy ryzyka z brakującymi polami."""
    data = {
        'symbol': 'EURUSD',
        'type': 'BUY'
    }
    
    with pytest.raises(ValueError, match="Brak wymaganego pola:"):
        TradingPrompts.get_risk_analysis_prompt(data)


def test_trading_prompts_risk_analysis_empty_data():
    """Test generowania promptu analizy ryzyka z pustymi danymi."""
    with pytest.raises(ValueError, match="Brak wymaganego pola:"):
        TradingPrompts.get_risk_analysis_prompt({})


def test_trading_prompts_risk_analysis_extreme_values():
    """Test generowania promptu analizy ryzyka z ekstremalnymi wartościami."""
    data = {
        'symbol': 'EURUSD',
        'type': 'BUY',
        'volume': 999999.99,
        'entry_price': 0.00001,
        'account_balance': 1000000000,
        'current_exposure': 0.9999
    }
    
    prompt = TradingPrompts.get_risk_analysis_prompt(data)
    assert isinstance(prompt, str)
    assert "999999.99" in prompt
    assert "0.0000" in prompt
    assert "1000000000.00" in prompt
    assert "99.99%" in prompt 