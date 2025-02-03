"""
Moduł obsługujący komunikację z platformą MetaTrader 5.
"""
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from src.utils.logger import TradingLogger

class MT5Handler:
    """Klasa obsługująca operacje na platformie MetaTrader 5."""
    
    TIMEFRAME_MAP = {
        '1M': mt5.TIMEFRAME_M1,
        '5M': mt5.TIMEFRAME_M5,
        '15M': mt5.TIMEFRAME_M15,
        '30M': mt5.TIMEFRAME_M30,
        '1H': mt5.TIMEFRAME_H1,
        '4H': mt5.TIMEFRAME_H4,
        '1D': mt5.TIMEFRAME_D1,
        '1W': mt5.TIMEFRAME_W1,
        '1MN': mt5.TIMEFRAME_MN1
    }

    def __init__(
        self, 
        symbol: str,
        timeframe: str = '1H',
        logger: Optional[TradingLogger] = None,
        strategy_name: str = "MT5Handler"
    ) -> None:
        """
        Inicjalizacja handlera MT5.

        Args:
            symbol: Symbol instrumentu (np. 'EURUSD')
            timeframe: Interwał czasowy ('1M', '5M', '15M', '30M', '1H', '4H', '1D', '1W', '1MN')
            logger: Logger do zapisywania operacji
            strategy_name: Nazwa strategii dla loggera
        
        Raises:
            RuntimeError: Gdy nie uda się zainicjalizować MT5
            ValueError: Gdy podano nieprawidłowy symbol lub timeframe
        """
        self.logger = logger or TradingLogger(strategy_name=strategy_name)

        if not mt5.initialize():
            self.logger.error(f"❌ Nie udało się zainicjalizować MT5: {mt5.last_error()}")
            raise RuntimeError("Nie udało się zainicjalizować MT5")
        
        # Sprawdź czy symbol istnieje
        if not mt5.symbol_info(symbol):
            self.logger.error(f"❌ Nieprawidłowy symbol: {symbol}")
            raise ValueError(f"Nieprawidłowy symbol: {symbol}")
        
        # Sprawdź czy timeframe jest prawidłowy
        if timeframe not in self.TIMEFRAME_MAP:
            self.logger.error(f"❌ Nieprawidłowy timeframe: {timeframe}")
            raise ValueError(f"Nieprawidłowy timeframe: {timeframe}")
        
        self.symbol = symbol
        self.timeframe = self.TIMEFRAME_MAP[timeframe]
        self.logger.info(f"🥷 MT5Handler zainicjalizowany dla {symbol} na timeframe {timeframe}")

    def _validate_volume(self, volume: float) -> bool:
        """
        Sprawdza czy wolumen jest prawidłowy dla danego symbolu.
        
        Args:
            volume: Wielkość pozycji w lotach
            
        Returns:
            bool: True jeśli wolumen jest prawidłowy
            
        Raises:
            ValueError: Gdy wolumen jest nieprawidłowy
        """
        if volume <= 0:
            raise ValueError("Wolumen musi być większy od 0")
            
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            raise ValueError(f"Nie można pobrać informacji o symbolu {self.symbol}")
            
        volume_min = float(symbol_info.volume_min)
        volume_max = float(symbol_info.volume_max)
        volume_step = float(symbol_info.volume_step)
            
        if volume < volume_min:
            raise ValueError(f"Wolumen poniżej minimum ({volume_min})")
            
        if volume > volume_max:
            raise ValueError(f"Wolumen powyżej maksimum ({volume_max})")
            
        # Sprawdź czy wolumen jest wielokrotnością kroku
        steps = round(volume / volume_step)
        if not np.isclose(volume, steps * volume_step, atol=1e-8):
            raise ValueError(f"Nieprawidłowy krok wolumenu (wymagany krok: {volume_step})")
            
        return True

    def _validate_sl_tp(self, direction: str, price: float, stop_loss: Optional[float], take_profit: Optional[float]) -> bool:
        """
        Sprawdza czy poziomy SL/TP są prawidłowe.
        
        Args:
            direction: Kierunek transakcji ('BUY' lub 'SELL')
            price: Cena otwarcia
            stop_loss: Poziom stop loss
            take_profit: Poziom take profit
            
        Returns:
            bool: True jeśli poziomy są prawidłowe
            
        Raises:
            ValueError: Gdy poziomy są nieprawidłowe
        """
        if stop_loss is not None:
            if direction == 'BUY' and stop_loss >= price:
                raise ValueError("Dla pozycji BUY, stop loss musi być poniżej ceny wejścia")
            elif direction == 'SELL' and stop_loss <= price:
                raise ValueError("Dla pozycji SELL, stop loss musi być powyżej ceny wejścia")
                
        if take_profit is not None:
            if direction == 'BUY' and take_profit <= price:
                raise ValueError("Dla pozycji BUY, take profit musi być powyżej ceny wejścia")
            elif direction == 'SELL' and take_profit >= price:
                raise ValueError("Dla pozycji SELL, take profit musi być poniżej ceny wejścia")
                
        return True

    async def open_position(
        self,
        direction: str,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Otwiera pozycję na rynku.

        Args:
            direction: Kierunek transakcji ('BUY' lub 'SELL')
            volume: Wielkość pozycji w lotach
            stop_loss: Poziom stop loss (opcjonalnie)
            take_profit: Poziom take profit (opcjonalnie)

        Returns:
            Dict zawierający status operacji i szczegóły
        """
        try:
            # Walidacja kierunku
            if direction not in ['BUY', 'SELL']:
                raise ValueError(f"Nieprawidłowy kierunek transakcji: {direction}")
            
            # Walidacja wolumenu
            try:
                self._validate_volume(volume)
            except ValueError as e:
                return {
                    "status": "error",
                    "message": str(e)
                }
            
            # Pobierz aktualną cenę
            price = await self.get_current_price()
            current_price = price['ask'] if direction == 'BUY' else price['bid']
            
            # Walidacja poziomów SL/TP
            try:
                self._validate_sl_tp(direction, current_price, stop_loss, take_profit)
            except ValueError as e:
                return {
                    "status": "error",
                    "message": str(e)
                }

            order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": order_type,
                "price": current_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            if stop_loss:
                request["sl"] = stop_loss
            if take_profit:
                request["tp"] = take_profit

            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"🥷 Otwarto pozycję {direction} na {self.symbol}, wolumen: {volume}")
                return {
                    "status": "success",
                    "volume": result.volume,
                    "price": result.price,
                    "comment": result.comment
                }
            else:
                self.logger.error(f"❌ Błąd podczas otwierania pozycji: {result.comment}")
                return {
                    "status": "error",
                    "message": result.comment,
                    "code": result.retcode
                }
        except Exception as e:
            self.logger.error(f"❌ Wyjątek podczas otwierania pozycji: {str(e)}")
            return {
                "status": "error",
                "message": f"Wyjątek: {str(e)}"
            }

    async def close_position(self) -> Dict[str, Any]:
        """
        Zamyka otwartą pozycję.

        Returns:
            Dict zawierający status operacji i szczegóły
        """
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                self.logger.warning("⚠️ Próba zamknięcia pozycji, ale brak otwartej pozycji")
                return {
                    "status": "error",
                    "message": "Brak otwartej pozycji"
                }

            position = positions[0]
            close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = await self.get_current_price()
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": position.ticket,
                "price": price['bid'] if position.type == mt5.ORDER_TYPE_BUY else price['ask'],
                "deviation": 20,
                "magic": 234000,
                "comment": "python script close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"🥷 Zamknięto pozycję na {self.symbol}, profit: {position.profit}")
                return {
                    "status": "success",
                    "volume": result.volume,
                    "price": result.price,
                    "profit": position.profit,
                    "comment": result.comment
                }
            else:
                self.logger.error(f"❌ Błąd podczas zamykania pozycji: {result.comment}")
                return {
                    "status": "error",
                    "message": f"Błąd: {result.comment}",
                    "code": result.retcode
                }
        except Exception as e:
            self.logger.error(f"❌ Wyjątek podczas zamykania pozycji: {str(e)}")
            return {
                "status": "error",
                "message": f"Wyjątek: {str(e)}"
            }

    async def get_current_price(self) -> Dict[str, float]:
        """
        Pobiera aktualną cenę instrumentu.

        Returns:
            Dict zawierający bid, ask i last price
        """
        tick = mt5.symbol_info_tick(self.symbol)
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "time": tick.time
        }

    async def get_historical_data(
        self,
        start_date: datetime,
        num_bars: int = 1000
    ) -> pd.DataFrame:
        """
        Pobiera dane historyczne.

        Args:
            start_date: Data początkowa
            num_bars: Liczba świec do pobrania

        Returns:
            DataFrame z danymi OHLCV

        Raises:
            RuntimeError: Gdy nie uda się pobrać danych
        """
        rates = mt5.copy_rates_from(self.symbol, self.timeframe, start_date, num_bars)
        
        if rates is None:
            self.logger.error(f"❌ Nie udało się pobrać danych historycznych dla {self.symbol}")
            raise RuntimeError("Nie udało się pobrać danych historycznych")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Zmiana nazw kolumn na standardowe
        df.rename(columns={
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        }, inplace=True)
        
        self.logger.info(f"🥷 Pobrano {len(df)} świec historycznych dla {self.symbol}")
        return df

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Pobiera informacje o koncie.

        Returns:
            Dict z informacjami o koncie
        """
        account = mt5.account_info()
        return {
            "login": account.login,
            "balance": account.balance,
            "equity": account.equity,
            "margin": account.margin,
            "margin_free": account.margin_free,
            "margin_level": account.margin_level,
            "currency": account.currency
        }

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Pobiera listę otwartych pozycji.

        Returns:
            Lista słowników z informacjami o pozycjach
        """
        positions = mt5.positions_get()
        if positions:
            return [{
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "open_price": pos.price_open,
                "current_price": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "swap": pos.swap,
                "time": datetime.fromtimestamp(pos.time)
            } for pos in positions]
        return []

    def cleanup(self):
        """Zamyka połączenie z MT5."""
        mt5.shutdown()
        self.logger.info("🥷 MT5Handler zamknięty")

    def __del__(self):
        """Destruktor - zamyka połączenie z MT5."""
        self.cleanup() 