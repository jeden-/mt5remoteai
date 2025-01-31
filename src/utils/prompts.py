"""
Moduł zawierający szablony promptów dla modeli AI.
"""
from typing import Dict, Any


def get_ollama_prompt_template() -> str:
    """
    Zwraca szablon promptu dla modelu Ollama.
    
    Returns:
        str: Szablon promptu
    """
    return """
    Analiza rynku dla {symbol}:
    - Aktualna cena: {current_price}
    - SMA 20: {sma_20}
    - SMA 50: {sma_50}
    
    Na podstawie powyższych danych:
    1. Określ aktualny trend (UP/DOWN/SIDEWAYS)
    2. Oceń siłę trendu (1-10)
    3. Podaj rekomendację (BUY/SELL/WAIT)
    4. Uzasadnij swoją decyzję
    
    Format odpowiedzi:
    TREND: [UP/DOWN/SIDEWAYS]
    STRENGTH: [1-10]
    RECOMMENDATION: [BUY/SELL/WAIT]
    REASONING: [uzasadnienie]
    """


def get_claude_prompt_template() -> str:
    """
    Zwraca szablon promptu dla modelu Claude.
    
    Returns:
        str: Szablon promptu
    """
    return """
    Przeprowadź szczegółową analizę rynku dla {symbol}:
    
    DANE RYNKOWE:
    - Aktualna cena: {current_price}
    - SMA 20: {sma_20}
    - SMA 50: {sma_50}
    
    Proszę o:
    1. Analizę trendu i jego siły
    2. Rekomendację transakcyjną (long/short/neutral)
    3. Sugestię poziomów SL i TP (w pipsach)
    4. Ocenę ryzyka transakcji
    5. Uzasadnienie rekomendacji
    
    Odpowiedź powinna zawierać:
    - Rekomendację
    - Sugerowany SL
    - Sugerowany TP
    - Uzasadnienie
    """


def format_market_data_for_prompt(market_data: Dict[str, Any]) -> str:
    """
    Formatuje dane rynkowe do postaci tekstu dla promptu.
    
    Args:
        market_data: Słownik z danymi rynkowymi
        
    Returns:
        str: Sformatowany tekst z danymi
        
    Raises:
        ValueError: Gdy brakuje wymaganych danych
    """
    if not market_data or not isinstance(market_data, dict):
        raise ValueError("Market data must be a non-empty dictionary")
        
    if 'symbol' not in market_data or not market_data['symbol']:
        raise ValueError("Symbol is required in market data")
        
    template = """
    Symbol: {symbol}
    Cena: {current_price}
    SMA 20: {sma_20}
    SMA 50: {sma_50}
    Zmiana 24h: {price_change_24h}
    Wolumen 24h: {volume_24h}
    """
    
    # Zastąp brakujące wartości przez 'N/A'
    formatted_data = {k: market_data.get(k, 'N/A') for k in [
        'symbol',
        'current_price',
        'sma_20',
        'sma_50',
        'price_change_24h',
        'volume_24h'
    ]}
    
    # Zamień None na 'N/A'
    for k, v in formatted_data.items():
        if v is None:
            formatted_data[k] = 'N/A'
            
    return template.format(**formatted_data)


class TradingPrompts:
    """Klasa zawierająca szablony promptów dla analizy rynku."""

    @staticmethod
    def get_market_analysis_prompt(data: Dict) -> str:
        """
        Generuje prompt do analizy rynku.

        Args:
            data: Słownik z danymi rynkowymi

        Returns:
            str: Wygenerowany prompt

        Raises:
            ValueError: Gdy brakuje wymaganych pól w danych
        """
        required_fields = ['symbol', 'current_price', 'sma_20', 'sma_50', 'price_change_24h', 'volume_24h']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Brak wymaganego pola: {field}")
                
        return f"""Analiza rynku dla {data['symbol']}:

DANE TECHNICZNE:
- Aktualna cena: {data['current_price']:.4f}
- SMA20: {data['sma_20']:.4f}
- SMA50: {data['sma_50']:.4f}
- Zmiana 24h: {data['price_change_24h']:.2f}%
- Wolumen 24h: {data['volume_24h']}

Wymagam konkretnej odpowiedzi zawierającej:
1. Kierunek trendu (UP/DOWN/SIDEWAYS)
2. Siła trendu (1-10)
3. Rekomendacja (BUY/SELL/WAIT)
4. Sugerowany SL (w pips)
5. Sugerowany TP (w pips)

Format odpowiedzi:
TREND_DIRECTION: [kierunek]
TREND_STRENGTH: [siła]
RECOMMENDATION: [rekomendacja]
SL_PIPS: [liczba]
TP_PIPS: [liczba]
REASONING: [krótkie uzasadnienie]."""
    
    @staticmethod
    def get_risk_analysis_prompt(data: Dict) -> str:
        """
        Generuje prompt do analizy ryzyka.

        Args:
            data: Słownik z parametrami transakcji

        Returns:
            str: Wygenerowany prompt

        Raises:
            ValueError: Gdy brakuje wymaganych pól w danych
        """
        required_fields = ['symbol', 'type', 'volume', 'entry_price', 'account_balance', 'current_exposure']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Brak wymaganego pola: {field}")
                
        return f"""Analiza ryzyka dla planowanej transakcji:

PARAMETRY:
- Instrument: {data['symbol']}
- Typ: {data['type']}
- Wielkość pozycji: {data['volume']:.2f}
- Cena wejścia: {data['entry_price']:.4f}
- Saldo konta: {data['account_balance']:.2f}
- Aktualna ekspozycja: {data['current_exposure']:.2%}

Wymagana odpowiedź:
1. Ocena ryzyka (LOW/MEDIUM/HIGH)
2. Risk/Reward Ratio
3. % kapitału na ryzyku
4. Rekomendacja (PROCEED/ADJUST/ABORT)

Format odpowiedzi:
RISK_LEVEL: [poziom]
RR_RATIO: [liczba]
CAPITAL_AT_RISK: [procent]
RECOMMENDATION: [rekomendacja]
REASONING: [krótkie uzasadnienie].""" 