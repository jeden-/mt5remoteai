"""
Moduł zawierający wspólne fixtures dla testów.
"""
import pytest
import os
from datetime import datetime


@pytest.fixture
def sample_market_data():
    """
    Fixture dostarczający przykładowe dane rynkowe.
    
    Returns:
        dict: Słownik z danymi rynkowymi
    """
    return {
        'symbol': 'EURUSD',
        'current_price': 1.1000,
        'sma_20': 1.0990,
        'sma_50': 1.0980,
        'price_change_24h': 0.5,
        'volume_24h': 10000,
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_trade_info():
    """
    Fixture dostarczający przykładowe dane transakcji.
    
    Returns:
        dict: Słownik z danymi transakcji
    """
    return {
        'symbol': 'EURUSD',
        'type': 'BUY',
        'volume': 0.1,
        'price': 1.1000,
        'sl': 1.0950,
        'tp': 1.1100,
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_ai_analysis():
    """
    Fixture dostarczający przykładową analizę AI.
    
    Returns:
        dict: Słownik z analizą AI
    """
    return {
        'symbol': 'EURUSD',
        'timestamp': datetime.now().isoformat(),
        'ollama_analysis': {
            'trend': 'UP',
            'strength': 8,
            'recommendation': 'BUY'
        },
        'claude_analysis': {
            'recommendation': 'LONG',
            'confidence': 0.85,
            'risk_level': 'MEDIUM'
        }
    }


@pytest.fixture
def sample_error_info():
    """
    Fixture dostarczający przykładowe informacje o błędzie.
    
    Returns:
        dict: Słownik z informacjami o błędzie
    """
    return {
        'type': 'CONNECTION_ERROR',
        'message': 'Nie można połączyć z MT5',
        'timestamp': datetime.now().isoformat(),
        'details': {
            'attempt': 3,
            'last_error': 'Connection timeout'
        }
    }


@pytest.fixture
def sample_strategy_config():
    """
    Fixture dostarczający przykładową konfigurację strategii.
    
    Returns:
        dict: Słownik z konfiguracją strategii
    """
    return {
        'max_position_size': 0.1,
        'max_risk_per_trade': 0.02,
        'allowed_symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
        'timeframe': 'H1',
        'indicators': {
            'sma_periods': [20, 50],
            'rsi_period': 14
        }
    }


@pytest.fixture
def sample_account_info():
    """
    Fixture dostarczający przykładowe informacje o koncie.
    
    Returns:
        dict: Słownik z informacjami o koncie
    """
    return {
        'balance': 10000,
        'equity': 10000,
        'margin': 0,
        'margin_level': 0,
        'profit': 0,
        'currency': 'USD',
        'leverage': 100
    } 