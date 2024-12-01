import numpy as np
from functools import lru_cache
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging

class TechnicalIndicators:
    def __init__(self, data):
        """Инициализация с DataFrame, содержащим OHLCV данные"""
        self.data = data.copy()
        
    def calculate_all_indicators(self):
        """Расчет всех технических индикаторов"""
        try:
            # RSI
            self.data['rsi'] = RSIIndicator(
                close=self.data['close'], 
                window=14
            ).rsi()
            
            # Bollinger Bands
            bb = BollingerBands(
                close=self.data['close'], 
                window=20, 
                window_dev=2
            )
            self.data['bb_high'] = bb.bollinger_hband()
            self.data['bb_mid'] = bb.bollinger_mavg()
            self.data['bb_low'] = bb.bollinger_lband()
            
            # Расчет процентной позиции цены в полосах Боллинджера
            self.data['bb_pct'] = (self.data['close'] - self.data['bb_low']) / (self.data['bb_high'] - self.data['bb_low']) - 0.5
            
            # Тренд
            self.data['sma20'] = SMAIndicator(
                close=self.data['close'], 
                window=20
            ).sma_indicator()
            
            self.data['sma50'] = SMAIndicator(
                close=self.data['close'], 
                window=50
            ).sma_indicator()
            
            # Определение направления тренда
            self.data['trend_direction'] = np.where(
                self.data['sma20'] > self.data['sma50'], 
                1,  # Восходящий тренд
                np.where(
                    self.data['sma20'] < self.data['sma50'], 
                    -1,  # Нисходящий тренд
                    0   # Боковой тренд
                )
            )
            
            # Расчет силы тренда
            self.data['trend_strength'] = abs(
                (self.data['sma20'] - self.data['sma50']) / self.data['sma50']
            )
            
            # Заполняем пропущенные значения
            self.data = self.data.fillna(method='bfill')
            
            return self.data
            
        except Exception as e:
            logging.error(f"Ошибка при расчете индикаторов: {str(e)}")
            raise