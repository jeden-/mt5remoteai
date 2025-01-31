"""
Moduł zawierający implementację podstawowej strategii tradingowej.
"""
from typing import Dict, Any, Optional
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy


class BasicStrategy(BaseStrategy):
    """Podstawowa strategia tradingowa łącząca analizę techniczną z AI."""

    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """
        Implementacja prostej strategii analizy rynku łączącej MT5, Ollama i Claude.
        
        Args:
            symbol: Symbol instrumentu
            
        Returns:
            Dict z wynikami analizy
        """
        try:
            # Pobierz dane rynkowe
            rates = self.mt5.get_rates(symbol, 'H1', 100)
            df = pd.DataFrame(rates)
            
            # Oblicz wskaźniki techniczne
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Przygotuj dane do analizy
            market_data = {
                'symbol': symbol,
                'current_price': float(df['close'].iloc[-1]),
                'sma_20': float(df['sma_20'].iloc[-1]),
                'sma_50': float(df['sma_50'].iloc[-1]),
                'price_change_24h': float((df['close'].iloc[-1] / df['close'].iloc[-24] - 1) * 100),
                'volume_24h': float(df['volume'].iloc[-24:].sum())
            }
            
            # Analizy AI
            ollama_analysis = await self.ollama.analyze_market_data(
                market_data,
                self.config.get('ollama_prompt_template', '')
            )
            
            claude_analysis = await self.claude.analyze_market_conditions(
                market_data,
                self.config.get('claude_prompt_template', '')
            )
            
            return {
                'market_data': market_data,
                'technical_indicators': {
                    'sma_20': market_data['sma_20'],
                    'sma_50': market_data['sma_50']
                },
                'ollama_analysis': ollama_analysis,
                'claude_analysis': claude_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas analizy rynku: {str(e)}")
            raise
    
    async def generate_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generuje sygnały na podstawie analizy.
        
        Args:
            analysis: Wyniki analizy rynku
            
        Returns:
            Dict z sygnałami
            
        Raises:
            ValueError: Gdy format danych jest nieprawidłowy
        """
        try:
            # Sprawdź wymagane pola
            if not isinstance(analysis, dict):
                raise ValueError("Analiza musi być słownikiem")
                
            required_fields = ['market_data', 'ollama_analysis', 'claude_analysis']
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Brak wymaganego pola: {field}")
                    
            if not isinstance(analysis['market_data'], dict):
                raise ValueError("market_data musi być słownikiem")
                
            if not isinstance(analysis['ollama_analysis'], str) or not isinstance(analysis['claude_analysis'], str):
                raise ValueError("Wyniki analizy AI muszą być stringami")
            
            # Parsuj wyniki analizy Ollama
            ollama_lines = analysis['ollama_analysis'].split('\n')
            ollama_dict = {}
            for line in ollama_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    ollama_dict[key.strip()] = value.strip()
            
            # Parsuj wyniki analizy Claude
            claude_lines = analysis['claude_analysis'].split('\n')
            claude_dict = {}
            for line in claude_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    claude_dict[key.strip()] = value.strip()
            
            # Sprawdź czy udało się sparsować wyniki
            if not ollama_dict or not claude_dict:
                raise ValueError("Nie udało się sparsować wyników analizy AI")
            
            # Generuj sygnały
            action = 'WAIT'
            if ollama_dict.get('RECOMMENDATION') == 'BUY' and 'long' in claude_dict.get('Rekomendacja', '').lower():
                action = 'BUY'
            elif ollama_dict.get('RECOMMENDATION') == 'SELL' and 'short' in claude_dict.get('Rekomendacja', '').lower():
                action = 'SELL'
            
            # Oblicz poziomy
            current_price = analysis['market_data']['current_price']
            sl_pips = float(claude_dict.get('Sugerowany SL', '20').split()[0])
            tp_pips = float(claude_dict.get('Sugerowany TP', '60').split()[0])
            
            pip_value = 0.0001 if 'JPY' not in analysis['market_data']['symbol'] else 0.01
            
            return {
                'symbol': analysis['market_data']['symbol'],
                'action': action,
                'entry_price': current_price,
                'stop_loss': current_price - (sl_pips * pip_value) if action == 'BUY' else current_price + (sl_pips * pip_value),
                'take_profit': current_price + (tp_pips * pip_value) if action == 'BUY' else current_price - (tp_pips * pip_value),
                'volume': self.config['max_position_size']
            }
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas generowania sygnałów: {str(e)}")
            raise
    
    async def execute_signals(self, signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Wykonuje sygnały tradingowe.
        
        Args:
            signals: Sygnały do wykonania
            
        Returns:
            Dict z informacjami o wykonanej transakcji lub None
        """
        try:
            if signals['action'] == 'WAIT':
                return None
                
            # Sprawdź czy symbol jest dozwolony
            if signals['symbol'] not in self.config['allowed_symbols']:
                logger.warning(f"⚠️ Symbol {signals['symbol']} nie jest dozwolony")
                return None
            
            # Złóż zlecenie
            order = self.mt5.place_order(
                symbol=signals['symbol'],
                order_type=signals['action'],
                volume=signals['volume'],
                price=signals['entry_price'],
                sl=signals['stop_loss'],
                tp=signals['take_profit']
            )
            
            if order:
                # Zapisz transakcję w bazie
                self.db.save_trade({
                    'ticket': order['ticket'],
                    'symbol': order['symbol'],
                    'type': order['type'],
                    'volume': order['volume'],
                    'price': order['price'],
                    'sl': order['sl'],
                    'tp': order['tp']
                })
                
                return order
                
        except Exception as e:
            logger.error(f"❌ Błąd podczas wykonywania zlecenia: {str(e)}")
            return None 