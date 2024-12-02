import numpy as np
from typing import Dict, List, Optional
import pandas as pd
import logging


class GOATStrategy:
    def __init__(self, risk_reward_ratio: float, max_loss: float):
        self.risk_reward_ratio = risk_reward_ratio
        self.max_loss = max_loss
        self.symbols = []
        self.positions = {}
        self.asset_metrics = {}

    def analyze_market(self, client) -> List[str]:
        """Анализирует топ-100 активов и выбирает лучшие для торговли"""
        try:
            # Получаем информацию о рынках
            markets = client.get_exchange_info()['symbols']
            tickers = client.get_ticker()
            
            logging.info(f"Получено {len(markets)} торговых пар")
            
            assets_data = []
            for market in markets:
                symbol = market['symbol']
                if not self._is_valid_symbol(symbol):
                    continue
                    
                ticker = next((t for t in tickers if t['symbol'] == symbol), None)
                if not ticker:
                    continue

                # Собираем метрики
                metrics = self._calculate_asset_metrics(ticker, market)
                if metrics:
                    assets_data.append(metrics)

            logging.info(f"Собраны метрики для {len(assets_data)} активов")
            return self._select_best_assets(assets_data)
            
        except Exception as e:
            logging.error(f"Ошибка при анализе рынка: {str(e)}")
            return []

    def _is_valid_symbol(self, symbol: str) -> bool:
        """Проверяет, подходит ли символ под критерии"""
        from config import ASSET_SELECTION
        
        try:
            # Проверяем формат символа
            if not symbol.endswith(ASSET_SELECTION['quote_currency']):
                return False
            
            # Получаем базовую валюту
            base = symbol.replace(ASSET_SELECTION['quote_currency'], '')
            
            # Проверяем, не в черном ли списке
            if base in ASSET_SELECTION['blacklist']:
                return False
            
            # Дополнительные проверки для стейблкоинов
            stablecoin_patterns = ['USD', 'PAX', 'DAI', 'TUSD', 'USDC', 'USDT', 'BUSD']
            if any(pattern in base for pattern in stablecoin_patterns):
                return False
            
            return True
        
        except Exception as e:
            logging.error(f"Ошибка проверки символа {symbol}: {str(e)}")
            return False

    def _calculate_asset_metrics(self, ticker: Dict, market: Dict) -> Optional[Dict]:
        """Рассчитывает метрики для актива"""
        try:
            volume_24h = float(ticker['quoteVolume'])
            price = float(ticker['lastPrice'])
            high_24h = float(ticker['highPrice'])
            low_24h = float(ticker['lowPrice'])
            
            # Волатильность
            volatility = (high_24h - low_24h) / low_24h if low_24h > 0 else 0
            
            # Тренд
            trend_strength = (price - low_24h) / (high_24h - low_24h) if (high_24h - low_24h) > 0 else 0
            
            # Объем
            volume_score = volume_24h / 1000000  # нормализация в миллионах
            
            return {
                'symbol': market['symbol'],
                'volatility': volatility,
                'volume_score': volume_score,
                'trend_strength': trend_strength,
                'price': price,
                'volume_24h': volume_24h
            }
        except (KeyError, ZeroDivisionError, ValueError) as e:
            logging.debug(f"Ошибка расчета метрик для {market.get('symbol', 'Unknown')}: {str(e)}")
            return None

    def _select_best_assets(self, assets_data: List[Dict]) -> List[str]:
        """Выбирает лучшие активы на основе метрик"""
        from config import ASSET_SELECTION
        
        try:
            if not assets_data:
                logging.warning("Нет данных для выбора активов")
                return []
            
            df = pd.DataFrame(assets_data)
            
            # Применяем фильтры
            df = df[
                (df['volume_24h'] >= ASSET_SELECTION['min_volume_24h']) &
                (df['price'] >= ASSET_SELECTION['min_price'])
            ]
            
            if df.empty:
                logging.warning("Нет активов, соответствующих критериям фильтрации")
                return []
            
            # Нормализуем метрики
            for col in ['volatility', 'volume_score', 'trend_strength']:
                if df[col].max() == df[col].min():
                    df[f'{col}_norm'] = 1.0
                else:
                    df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            
            # Рассчитываем общий скор
            weights = ASSET_SELECTION['selection_criteria']
            df['total_score'] = (
                df['volatility_norm'] * weights['volatility_weight'] +
                df['volume_score_norm'] * weights['volume_weight'] +
                df['trend_strength_norm'] * weights['trend_weight']
            )
            
            # Выбираем лучшие активы
            max_assets = min(
                ASSET_SELECTION['selection_criteria']['max_selected_assets'],
                len(df)
            )
            
            best_assets = df.nlargest(max_assets, 'total_score')
            
            # Обновляем список символов и метрики
            self.symbols = best_assets['symbol'].tolist()
            self.asset_metrics = best_assets.set_index('symbol').to_dict('index')
            
            # Обновляем распределение средств
            self._update_allocation()
            
            logging.info(f"Выбрано {len(self.symbols)} активов: {', '.join(self.symbols)}")
            return self.symbols
            
        except Exception as e:
            logging.error(f"Ошибка при выборе активов: {str(e)}")
            return []

    def _update_allocation(self):
        """Обновляет распределение средств между активами"""
        from config import TRADING_PARAMS
        
        if not self.symbols:
            return
            
        # Равное распределение
        allocation_per_asset = 1.0 / len(self.symbols)
        
        TRADING_PARAMS['allocation'] = {
            symbol: allocation_per_asset for symbol in self.symbols
        }

    def generate_signal(self, data, symbol):
        """
        Args:
            data (list): Исторические данные
            symbol (str): Торговая пара
        """
        last_close = data[-1]['close']
        moving_average_50 = np.mean([d['close'] for d in data[-50:]])

        if last_close > moving_average_50:
            return {"symbol": symbol, "signal": "buy"}
        elif last_close < moving_average_50:
            return {"symbol": symbol, "signal": "sell"}
        else:
            return {"symbol": symbol, "signal": "hold"}

    def calculate_stop_loss(self, entry_price):
        return entry_price * (1 - self.max_loss)

    def calculate_take_profit(self, entry_price):
        return entry_price * (1 + self.risk_reward_ratio * self.max_loss)

    def calculate_position_size(self, entry_price, balance, symbol):
        """
        Args:
            entry_price (float): Цена входа
            balance (float): Баланс счета
            symbol (str): Торговая пара
        """
        # Распределяем риск между всеми активами
        max_risk_per_trade = (balance * 0.01) / len(self.symbols)
        
        stop_loss_price = self.calculate_stop_loss(entry_price)
        stop_loss_amount = abs(entry_price - stop_loss_price)
        
        if stop_loss_amount == 0:
            return 0
            
        position_size = max_risk_per_trade / stop_loss_amount
        
        return position_size

    async def initialize(self, exchange):
        """Инициализация стратегии и первичный анализ рынка"""
        self.symbols = await self.analyze_market(exchange)
        if not self.symbols:
            raise ValueError("Не удалось найти подходящие активы для торговли")
        return self.symbols
