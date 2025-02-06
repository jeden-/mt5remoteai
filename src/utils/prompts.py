"""
Moduł zawierający szablony promptów i funkcje pomocnicze do formatowania danych dla modeli AI
"""
from typing import Dict, Union, Any
from decimal import Decimal
import json

def get_ollama_prompt_template() -> str:
    """Zwraca szablon promptu dla modelu Ollama."""
    return """
    Analiza techniczna dla {symbol}:
    Aktualna cena: {current_price}
    SMA 20: {sma_20}
    SMA 50: {sma_50}

    Proszę o analizę w następującym formacie:
    TREND: [UP/DOWN/SIDEWAYS]
    STRENGTH: [WEAK/MODERATE/STRONG]
    RECOMMENDATION: [BUY/SELL/HOLD]
    REASONING: [Szczegółowe uzasadnienie]
    """

def get_claude_prompt_template() -> str:
    """Zwraca szablon promptu dla modelu Claude."""
    return """
    Analiza techniczna dla {symbol}:
    Aktualna cena: {current_price}
    SMA 20: {sma_20}
    SMA 50: {sma_50}

    Proszę o analizę zawierającą:
    1. Rekomendację [BUY/SELL/HOLD]
    2. Sugerowany SL (w pipsach)
    3. Sugerowany TP (w pipsach)
    4. Uzasadnienie decyzji
    """

def format_market_data_for_prompt(market_data: Dict[str, Any]) -> str:
    """
    Formatuje dane rynkowe do postaci tekstu dla promptu.
    
    Args:
        market_data: Słownik zawierający dane rynkowe
        
    Returns:
        Sformatowany tekst z danymi rynkowymi
        
    Raises:
        ValueError: Gdy dane wejściowe są nieprawidłowe
    """
    if not isinstance(market_data, dict) or not market_data:
        raise ValueError("Market data must be a non-empty dictionary")
    
    if 'symbol' not in market_data:
        raise ValueError("Symbol is required in market data")
        
    # Konwersja wartości None na "N/A"
    formatted_data = {
        'symbol': market_data['symbol'],
        'current_price': "N/A" if market_data.get('current_price') is None else str(market_data.get('current_price')),
        'sma_20': "N/A" if market_data.get('sma_20') is None else str(market_data.get('sma_20')),
        'sma_50': "N/A" if market_data.get('sma_50') is None else str(market_data.get('sma_50')),
        'price_change_24h': "N/A" if market_data.get('price_change_24h') is None else str(market_data.get('price_change_24h')),
        'volume_24h': "N/A" if market_data.get('volume_24h') is None else str(market_data.get('volume_24h'))
    }
    
    return f"""
    Symbol: {formatted_data['symbol']}
    Current Price: {formatted_data['current_price']}
    SMA 20: {formatted_data['sma_20']}
    SMA 50: {formatted_data['sma_50']}
    24h Price Change: {formatted_data['price_change_24h']}
    24h Volume: {formatted_data['volume_24h']}
    """

class TradingPrompts:
    """Klasa zawierająca metody do generowania promptów tradingowych"""
    
    REQUIRED_MARKET_FIELDS = ['symbol', 'current_price', 'sma_20', 'sma_50', 'price_change_24h', 'volume_24h']
    REQUIRED_RISK_FIELDS = ['symbol', 'type', 'volume', 'entry_price', 'account_balance', 'current_exposure']
    
    @classmethod
    def get_market_analysis_prompt(cls, data: Dict[str, Any]) -> str:
        """
        Generuje prompt do analizy rynku.
        
        Args:
            data: Słownik z danymi rynkowymi
            
        Returns:
            Tekst promptu
            
        Raises:
            ValueError: Gdy brakuje wymaganych pól
        """
        for field in cls.REQUIRED_MARKET_FIELDS:
            if field not in data:
                raise ValueError(f"Brak wymaganego pola: {field}")
        
        # Konwersja wartości na string z odpowiednim formatowaniem
        formatted_data = {
            'symbol': str(data['symbol']),
            'current_price': f"{float(data['current_price']):.4f}",
            'sma_20': f"{float(data['sma_20']):.4f}",
            'sma_50': f"{float(data['sma_50']):.4f}",
            'price_change_24h': f"{float(data['price_change_24h']):.2f}",
            'volume_24h': str(data['volume_24h'])
        }
        
        return f"""
        MARKET ANALYSIS FOR {formatted_data['symbol']}
        
        CURRENT DATA:
        Price: {formatted_data['current_price']}
        SMA 20: {formatted_data['sma_20']}
        SMA 50: {formatted_data['sma_50']}
        24h Change: {formatted_data['price_change_24h']}%
        24h Volume: {formatted_data['volume_24h']}
        
        PLEASE PROVIDE:
        TREND_DIRECTION: [UP/DOWN/SIDEWAYS]
        TREND_STRENGTH: [WEAK/MODERATE/STRONG]
        RECOMMENDATION: [BUY/SELL/HOLD]
        SL_PIPS: [NUMBER]
        TP_PIPS: [NUMBER]
        REASONING: [Detailed explanation]
        """
    
    @classmethod
    def get_risk_analysis_prompt(cls, data: Dict[str, Any]) -> str:
        """
        Generuje prompt do analizy ryzyka.
        
        Args:
            data: Słownik z danymi o pozycji
            
        Returns:
            Tekst promptu
            
        Raises:
            ValueError: Gdy brakuje wymaganych pól
        """
        for field in cls.REQUIRED_RISK_FIELDS:
            if field not in data:
                raise ValueError(f"Brak wymaganego pola: {field}")
        
        # Konwersja wartości na string z odpowiednim formatowaniem
        formatted_data = {
            'symbol': str(data['symbol']),
            'type': str(data['type']),
            'volume': f"{float(data['volume']):.2f}",
            'entry_price': f"{float(data['entry_price']):.4f}",
            'account_balance': f"{float(data['account_balance']):.2f}",
            'current_exposure': f"{float(data['current_exposure'])*100:.2f}"
        }
        
        return f"""
        RISK ANALYSIS FOR {formatted_data['symbol']}
        
        POSITION DETAILS:
        Type: {formatted_data['type']}
        Volume: {formatted_data['volume']}
        Entry Price: {formatted_data['entry_price']}
        Account Balance: {formatted_data['account_balance']}
        Current Exposure: {formatted_data['current_exposure']}%
        
        PLEASE PROVIDE:
        RISK_LEVEL: [LOW/MEDIUM/HIGH]
        RR_RATIO: [NUMBER]
        CAPITAL_AT_RISK: [PERCENTAGE]
        RECOMMENDATION: [PROCEED/ADJUST/REJECT]
        REASONING: [Detailed explanation]
        """ 