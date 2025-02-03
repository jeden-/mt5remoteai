"""
Moduł zawierający implementację podstawowej strategii tradingowej.
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from loguru import logger
from decimal import Decimal

from .base_strategy import BaseStrategy
from src.utils.technical_indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands


class BasicStrategy(BaseStrategy):
    """Podstawowa strategia tradingowa łącząca analizę techniczną z AI."""

    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """
        Analizuje rynek dla danego symbolu.

        Args:
            symbol: Symbol do analizy

        Returns:
            Dict z wynikami analizy

        Raises:
            Exception: W przypadku błędu analizy
        """
        try:
            # Pobierz dane historyczne
            df = await self.mt5_connector.get_rates(symbol, 100)
            
            # Sprawdź czy mamy wystarczająco danych
            if df.empty or len(df) < 50:
                raise ValueError("Brak danych")
                
            # Oblicz wskaźniki techniczne
            sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
            
            # Przygotuj dane rynkowe
            market_data = {
                'current_price': float(df['close'].iloc[-1]),
                'volume': float(df['volume'].iloc[-1]),
                'trend': 'up' if sma_20 > sma_50 else 'down'
            }
            
            # Analizy AI
            ollama_analysis = await self.ollama_connector.analyze_market_data(df)
            claude_analysis = await self.anthropic_connector.analyze_market_conditions(df)
            
            return {
                'technical_indicators': {
                    'sma_20': sma_20,
                    'sma_50': sma_50
                },
                'market_data': market_data,
                'ollama_analysis': ollama_analysis,
                'claude_analysis': claude_analysis
            }
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas analizy rynku: {str(e)}")
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
                
            # Sprawdź czy mamy wszystkie potrzebne dane
            market_data = analysis['market_data']
            ollama_analysis = analysis['ollama_analysis']
            claude_analysis = analysis['claude_analysis']

            if not all(isinstance(x, dict) for x in [ollama_analysis, claude_analysis]):
                raise ValueError("Wyniki analizy AI muszą być słownikami")
            
            # Generuj sygnały
            signals = []
            
            # Logika generowania sygnałów...
            
            return {
                'signals': signals,
                'analysis_summary': {
                    'market_data': market_data,
                    'ai_recommendations': {
                        'ollama': ollama_analysis.get('recommendation'),
                        'claude': claude_analysis.get('recommendation')
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Błąd podczas generowania sygnałów: {str(e)}")
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
            if signals.get('action') == 'WAIT':
                return None
                
            # Sprawdź czy symbol jest dozwolony
            if signals.get('symbol') not in self.config.get('allowed_symbols', []):
                logger.warning(f"⚠️ Symbol {signals.get('symbol')} nie jest dozwolony")
                return None
            
            # Złóż zlecenie
            order = await self.mt5.place_order(
                symbol=signals['symbol'],
                order_type=signals['action'],
                volume=signals['volume'],
                price=signals['entry_price'],
                sl=signals['stop_loss'],
                tp=signals['take_profit']
            )
            
            if order:
                # Zapisz transakcję w bazie
                trade_data = {
                    'ticket': order['ticket'],
                    'symbol': order['symbol'],
                    'type': order['type'],
                    'volume': order['volume'],
                    'price': order['price'],
                    'sl': order['sl'],
                    'tp': order['tp']
                }
                await self.db.save_trade(trade_data)
                return order
                
            return None
                
        except Exception as e:
            logger.error(f"❌ Błąd podczas wykonywania zlecenia: {str(e)}")
            return None

    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """
        Oblicza wielkość pozycji na podstawie ryzyka na trade.
        
        Args:
            symbol: Symbol instrumentu
            entry_price: Cena wejścia
            stop_loss: Poziom stop loss
            
        Returns:
            float: Wielkość pozycji w lotach
        """
        try:
            # Pobierz parametry ryzyka
            risk_per_trade = self.config.get('risk_per_trade', 0.02)  # Domyślnie 2% kapitału
            account_balance = self.config.get('account_balance', 10000.0)
            max_position_size = self.config.get('max_position_size', 1.0)
            
            # Oblicz kwotę ryzyka
            risk_amount = account_balance * risk_per_trade
            
            # Oblicz wielkość pozycji
            pip_value = 0.0001 if 'JPY' not in symbol else 0.01
            stop_loss_pips = abs(entry_price - stop_loss) / pip_value
            position_size = risk_amount / (stop_loss_pips * 10)  # 10$ na pip dla 1 lota
            
            # Ogranicz wielkość pozycji
            position_size = min(position_size, max_position_size)
            
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania wielkości pozycji: {str(e)}")
            return self.config.get('max_position_size', 1.0)
    
    def _calculate_stop_loss(self, symbol: str, direction: str, entry_price: float) -> float:
        """
        Oblicza poziom stop loss.
        
        Args:
            symbol: Symbol instrumentu
            direction: Kierunek pozycji ('BUY' lub 'SELL')
            entry_price: Cena wejścia
            
        Returns:
            float: Poziom stop loss
        """
        try:
            stop_loss_pips = self.config.get('stop_loss_pips', 50)
            pip_value = 0.0001 if 'JPY' not in symbol else 0.01
            pip_distance = stop_loss_pips * pip_value
            
            if direction == 'BUY':
                return float(round(Decimal(str(entry_price)) - Decimal(str(pip_distance)), 5))
            else:
                return float(round(Decimal(str(entry_price)) + Decimal(str(pip_distance)), 5))
                
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania stop loss: {str(e)}")
            return entry_price
    
    def _calculate_take_profit(self, symbol: str, direction: str, entry_price: float) -> float:
        """
        Oblicza poziom take profit.
        
        Args:
            symbol: Symbol instrumentu
            direction: Kierunek pozycji ('BUY' lub 'SELL')
            entry_price: Cena wejścia
            
        Returns:
            float: Poziom take profit
        """
        try:
            take_profit_pips = self.config.get('take_profit_pips', 100)
            pip_value = 0.0001 if 'JPY' not in symbol else 0.01
            pip_distance = take_profit_pips * pip_value
            
            if direction == 'BUY':
                return float(round(Decimal(str(entry_price)) + Decimal(str(pip_distance)), 5))
            else:
                return float(round(Decimal(str(entry_price)) - Decimal(str(pip_distance)), 5))
                
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania take profit: {str(e)}")
            return entry_price

    def analyze_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizuje wskaźniki techniczne.

        Args:
            data: DataFrame z danymi rynkowymi

        Returns:
            Dict z wynikami analizy technicznej
        """
        try:
            # Przygotuj kopię danych
            df = data.copy()
            
            # Oblicz średnie kroczące
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # Oblicz RSI
            df['rsi'] = calculate_rsi(df['close'], 14)
            
            # Oblicz MACD
            macd_data = calculate_macd(df['close'])
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_hist'] = macd_data['hist']
            
            # Oblicz Bollinger Bands
            bb_data = calculate_bollinger_bands(df['close'], 20, 2)
            df['bb_upper'] = bb_data['upper']
            df['bb_middle'] = bb_data['middle']
            df['bb_lower'] = bb_data['lower']
            
            # Oblicz momentum
            df['momentum'] = df['close'].pct_change(10)
            
            # Oblicz ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # Przygotuj wyniki analizy
            latest = df.iloc[-1]
            
            analysis = {
                'trend': {
                    'sma_20': float(latest['sma_20']),
                    'sma_50': float(latest['sma_50']),
                    'sma_200': float(latest['sma_200']),
                    'trend_direction': 'UP' if latest['sma_20'] > latest['sma_50'] else 'DOWN',
                    'trend_strength': abs(float(latest['sma_20'] - latest['sma_50']))
                },
                'momentum': {
                    'rsi': float(latest['rsi']),
                    'macd': float(latest['macd']),
                    'macd_signal': float(latest['macd_signal']),
                    'macd_hist': float(latest['macd_hist']),
                    'momentum': float(latest['momentum'])
                },
                'volatility': {
                    'bb_upper': float(latest['bb_upper']),
                    'bb_middle': float(latest['bb_middle']),
                    'bb_lower': float(latest['bb_lower']),
                    'bb_width': float((latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']),
                    'atr': float(latest['atr'])
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas analizy wskaźników technicznych: {str(e)}")
            raise

    def calculate_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Oblicza metryki wydajności strategii.

        Args:
            trades: Lista transakcji

        Returns:
            Dict z metrykami wydajności
        """
        try:
            if not trades:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'average_win': 0.0,
                    'average_loss': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'recovery_factor': 0.0,
                    'profit_per_trade': 0.0
                }
            
            # Przygotuj dane
            profits = [t['profit'] for t in trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]
            
            # Oblicz podstawowe metryki
            total_trades = len(trades)
            winning_trades_count = len(winning_trades)
            win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
            
            # Oblicz średnie zyski/straty
            average_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
            average_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
            
            # Oblicz profit factor
            gross_profit = sum(winning_trades) if winning_trades else 0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Oblicz drawdown
            cumulative = np.cumsum(profits)
            max_drawdown = abs(min(cumulative - np.maximum.accumulate(cumulative)))
            
            # Oblicz Sharpe Ratio (zakładając bezryzykowną stopę 0%)
            returns = np.array(profits)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            
            # Oblicz Recovery Factor
            total_profit = sum(profits)
            recovery_factor = total_profit / max_drawdown if max_drawdown > 0 else 0
            
            return {
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'average_win': float(average_win),
                'average_loss': float(average_loss),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'recovery_factor': float(recovery_factor),
                'profit_per_trade': float(total_profit / total_trades) if total_trades > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas obliczania metryk wydajności: {str(e)}")
            raise

    def validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """
        Waliduje wyniki analizy technicznej.

        Args:
            analysis: Dict z wynikami analizy

        Returns:
            bool: True jeśli analiza jest prawidłowa
        """
        try:
            # Sprawdź czy wszystkie wymagane sekcje są obecne
            required_sections = ['trend', 'momentum', 'volatility']
            if not all(section in analysis for section in required_sections):
                logger.error("❌ Brak wymaganych sekcji w analizie")
                return False
            
            # Sprawdź sekcję trend
            trend = analysis['trend']
            required_trend = ['sma_20', 'sma_50', 'sma_200', 'trend_direction', 'trend_strength']
            if not all(field in trend for field in required_trend):
                logger.error("❌ Brak wymaganych pól w sekcji trend")
                return False
            
            # Sprawdź sekcję momentum
            momentum = analysis['momentum']
            required_momentum = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'momentum']
            if not all(field in momentum for field in required_momentum):
                logger.error("❌ Brak wymaganych pól w sekcji momentum")
                return False
            
            # Sprawdź sekcję volatility
            volatility = analysis['volatility']
            required_volatility = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr']
            if not all(field in volatility for field in required_volatility):
                logger.error("❌ Brak wymaganych pól w sekcji volatility")
                return False
            
            # Sprawdź wartości
            if not (0 <= momentum['rsi'] <= 100):
                logger.error("❌ Nieprawidłowa wartość RSI")
                return False
                
            if not (volatility['bb_lower'] <= volatility['bb_middle'] <= volatility['bb_upper']):
                logger.error("❌ Nieprawidłowe wartości Bollinger Bands")
                return False
                
            if volatility['atr'] <= 0:
                logger.error("❌ Nieprawidłowa wartość ATR")
                return False
                
            if volatility['bb_width'] <= 0:
                logger.error("❌ Nieprawidłowa szerokość Bollinger Bands")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Błąd podczas walidacji analizy: {str(e)}")
            return False 