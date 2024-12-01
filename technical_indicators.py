import numpy as np
from functools import lru_cache
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data
        self.config = self._load_indicator_config()
        self._cache = {}  # Кэш для хранения результатов
        
    def _load_indicator_config(self):
        """Загрузка конфигурации индикаторов"""
        try:
            from config import INDICATOR_PARAMS
            return INDICATOR_PARAMS
        except ImportError:
            return {
                'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
                'bollinger_bands': {'period': 20, 'std_dev': 2},
                'moving_averages': {'short_period': 20, 'long_period': 50}
            }
    
    def calculate_all_indicators(self):
        """Вычисляет все индикаторы"""
        try:
            # Базовые индикаторы
            self._calculate_moving_averages()
            self._calculate_momentum_indicators()
            self._calculate_volatility_indicators()
            self._calculate_trend_indicators()
            
            # Проверяем наличие всех необходимых индикаторов
            required_indicators = ['rsi', 'bb_high', 'bb_low', 'sma20', 'sma50', 'close']
            missing_indicators = [ind for ind in required_indicators if ind not in self.data.columns]
            
            if missing_indicators:
                print(f"Внимание: отсутствуют индикаторы: {missing_indicators}")
                
            return self.data
            
        except Exception as e:
            print(f"Ошибка при расчете индикаторов: {str(e)}")
            return self.data
    
    @lru_cache(maxsize=128)
    def _calculate_moving_averages(self):
        """Расчет всех скользящих средних с кэшированием"""
        ma_params = self.config['moving_averages']
        cache_key = f"ma_{ma_params['short_period']}_{ma_params['long_period']}"
        
        if cache_key not in self._cache:
            self._cache[cache_key] = {
                'sma20': self.calculate_sma(ma_params['short_period']),
                'sma50': self.calculate_sma(ma_params['long_period']),
                'ema20': self.calculate_ema(ma_params['short_period']),
                'ema50': self.calculate_ema(ma_params['long_period'])
            }
            
            # Обновляем DataFrame
            for key, value in self._cache[cache_key].items():
                self.data[key] = value
            
            # Рассчитываем пересечения
            self.data['ma_cross'] = np.where(
                self.data['sma20'] > self.data['sma50'], 1,
                np.where(self.data['sma20'] < self.data['sma50'], -1, 0)
            )
    
    @lru_cache(maxsize=128)
    def _calculate_momentum_indicators(self):
        """Расчет индикаторов импульса с кэшированием"""
        rsi_params = self.config['rsi']
        cache_key = f"momentum_{rsi_params['period']}"
        
        if cache_key not in self._cache:
            self._cache[cache_key] = {
                'rsi': self.calculate_rsi(rsi_params['period']),
                'momentum': self.data['close'].diff(periods=10)
            }
            
            # Обновляем DataFrame
            for key, value in self._cache[cache_key].items():
                self.data[key] = value
            
            # Рассчитываем RSI сигналы
            self.data['rsi_signal'] = np.where(
                self.data['rsi'] > rsi_params['overbought'], -1,
                np.where(self.data['rsi'] < rsi_params['oversold'], 1, 0)
            )
    
    @lru_cache(maxsize=128)
    def _calculate_volatility_indicators(self):
        """Расчет индикаторов волатильности с кэшированием"""
        bb_params = self.config['bollinger_bands']
        cache_key = f"bb_{bb_params['period']}_{bb_params['std_dev']}"
        
        if cache_key not in self._cache:
            bb = BollingerBands(
                close=self.data['close'],
                window=bb_params['period'],
                window_dev=bb_params['std_dev']
            )
            
            self._cache[cache_key] = {
                'bb_high': bb.bollinger_hband(),
                'bb_mid': bb.bollinger_mavg(),
                'bb_low': bb.bollinger_lband()
            }
            
            # Обновляем DataFrame
            for key, value in self._cache[cache_key].items():
                self.data[key] = value
            
            # Рассчитываем процент B
            self.data['bb_pct'] = (self.data['close'] - self.data['bb_low']) / (
                self.data['bb_high'] - self.data['bb_low']
            )
    
    @lru_cache(maxsize=128)
    def _calculate_trend_indicators(self):
        """Расчет индикаторов тренда с кэшированием"""
        if 'trend' not in self._cache:
            self._cache['trend'] = {
                'trend_direction': np.where(
                    self.data['sma20'] > self.data['sma50'], 1,
                    np.where(self.data['sma20'] < self.data['sma50'], -1, 0)
                ),
                'trend_strength': abs(
                    (self.data['sma20'] - self.data['sma50']) / self.data['sma50'] * 100
                )
            }
            
            # Обновляем DataFrame
            for key, value in self._cache['trend'].items():
                self.data[key] = value
    
    def clear_cache(self):
        """Очистка кэша"""
        self._cache.clear()
        # Очищаем также кэш декораторов lru_cache
        self.calculate_sma.cache_clear()
        self.calculate_ema.cache_clear()
        self.calculate_rsi.cache_clear()
    
    @lru_cache(maxsize=128)
    def calculate_sma(self, period):
        """Вычисляет простую скользящую среднюю"""
        return SMAIndicator(close=self.data['close'], window=period).sma_indicator()
    
    @lru_cache(maxsize=128)
    def calculate_ema(self, period):
        """Вычисляет экспоненциальную скользящую среднюю"""
        return EMAIndicator(close=self.data['close'], window=period).ema_indicator()
    
    @lru_cache(maxsize=128)
    def calculate_rsi(self, period=14):
        """Вычисляет индекс относительной силы (RSI)"""
        return RSIIndicator(close=self.data['close'], window=period).rsi()