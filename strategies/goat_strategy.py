import numpy as np
from typing import Dict, List, Optional
import pandas as pd
import logging
import config


class GOATStrategy:
    def __init__(self, risk_reward_ratio: float, max_loss: float, balance: float = None):
        self.risk_reward_ratio = risk_reward_ratio
        self.max_loss = max_loss
        self.balance = balance
        self.symbols = []
        self.positions = {}
        self.asset_metrics = {}
        self.logger = logging.getLogger('TradingBot')

    def analyze_market(self, client) -> List[str]:
        """Анализирует топ-100 активов и выбирает лучшие для торговли"""
        try:
            # Получаем информацию о рынках
            markets = client.get_exchange_info()['symbols']
            tickers = client.get_ticker()
            
            self.logger.info(f"Получено {len(markets)} торговых пар")
            
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

            self.logger.info(f"Собраны метрики для {len(assets_data)} активов")
            return self._select_best_assets(assets_data)
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе рынка: {str(e)}")
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
            
            # Дополнительные проверки ля стейблкоинов
            stablecoin_patterns = ['USD', 'PAX', 'DAI', 'TUSD', 'USDC', 'USDT', 'BUSD']
            if any(pattern in base for pattern in stablecoin_patterns):
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Ошибка проверки символа {symbol}: {str(e)}")
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
            self.logger.debug(f"Ошибка расчета метрик для {market.get('symbol', 'Unknown')}: {str(e)}")
            return None

    def _select_best_assets(self, assets_data: List[Dict]) -> List[str]:
        """Выбирает лучшие активы на основе метрик"""
        from config import ASSET_SELECTION
        
        try:
            if not assets_data:
                self.logger.warning("Нет данных для выбора активов")
                return []
            
            df = pd.DataFrame(assets_data)
            
            # Применяем фильтры
            df = df[
                (df['volume_24h'] >= ASSET_SELECTION['min_volume_24h']) &
                (df['price'] >= ASSET_SELECTION['min_price'])
            ]
            
            if df.empty:
                self.logger.warning("Нет активов, соответствующих критериям фильтрации")
                return []
            
            # Нормлизуем метрики
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
            
            # Логируем метрики выбранных активов
            for _, row in best_assets.iterrows():
                self.logger.info(
                    f"Актив {row['symbol']}: "
                    f"Волатильность={row['volatility']:.2%}, "
                    f"Объем=${row['volume_24h']/1000000:.1f}M, "
                    f"Тренд={row['trend_strength']:.2f}, "
                    f"Общий скор={row['total_score']:.2f}"
                )
            
            # Обновляем список символов и метрики
            self.symbols = best_assets['symbol'].tolist()
            self.asset_metrics = best_assets.set_index('symbol').to_dict('index')
            
            # Обновляем распределение средств
            self._update_allocation()
            
            return self.symbols
            
        except Exception as e:
            self.logger.error(f"Ошибка при выборе активов: {str(e)}")
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

    def calculate_position_size(self, entry_price, stop_loss, balance):
        """Расчет размера позиции с учетом ограничений"""
        try:
            if balance is None:
                self.logger.error("Баланс не определен")
                return 0
            
            # Рассчитываем размер позиции исходя из риска
            risk_amount = balance * (config.POSITION_SIZING['risk_per_trade'] / 100)
            position_size = risk_amount / abs(entry_price - stop_loss)
            
            # Рассчитываем стоимость позиции в USDT
            position_value = position_size * entry_price
            
            # Проверяем минимальный размер позиции
            if position_value < config.POSITION_SIZING['min_position_size']:
                self.logger.warning(f"Размер позиции меньше минимального: {position_value:.2f} USDT")
                return 0
            
            # Проверяем максимальный размер позиции
            max_position_value = min(
                balance * (config.POSITION_SIZING['max_position_percent'] / 100),
                config.POSITION_SIZING['max_position_size']
            )
            
            if position_value > max_position_value:
                position_size = max_position_value / entry_price
                self.logger.info(f"Размер позиции ограничен до {position_size:.8f} ({max_position_value:.2f} USDT)")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета размера позиции: {str(e)}")
            return 0

    async def initialize(self, exchange):
        """Инициализация стратегии и первичный анализ рынка"""
        self.symbols = await self.analyze_market(exchange)
        if not self.symbols:
            raise ValueError("Не удалось найти подходящие активы для торговли")
        return self.symbols

    def analyze(self, market_data):
        signals = []
        
        for symbol, data in market_data.items():
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Анализ символа: {symbol}")
                
                # Преобразуем данные в DataFrame
                df = pd.DataFrame(data)
                
                # Проверяем достаточность данных
                if len(df) < 20:
                    self.logger.info(f"Недостаточно свечей для {symbol}: {len(df)} (нужно минимум 20)")
                    continue
                    
                # Выводим последние цены
                self.logger.info(f"Последние цены:")
                for i in range(min(5, len(df))):
                    candle = df.iloc[-(i+1)]
                    self.logger.info(f"{i+1}. Цена: {candle['close']:.8f}, Объем: {candle['volume']:.2f}")
                
                if self._check_signal_conditions(df):
                    entry_price = float(df['close'].iloc[-1])
                    
                    # Расчёт стоп-лосса и тейк-профита
                    stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT/100)
                    take_profit = entry_price * (1 + config.TAKE_PROFIT_PERCENT/100)
                    
                    signals.append({
                        'symbol': symbol,
                        'direction': 'buy',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'stop_loss_percent': -config.STOP_LOSS_PERCENT,
                        'take_profit': take_profit,
                        'take_profit_percent': config.TAKE_PROFIT_PERCENT
                    })
                    self.logger.info(f"Сгенерирован сигнал для {symbol}")
                else:
                    self.logger.info(f"Нет сигнала для {symbol}")
                
                self.logger.info(f"{'='*50}\n")
                    
            except Exception as e:
                self.logger.error(f"Ошибка анализа {symbol}: {str(e)}")
                continue
        
        return signals

    def _check_signal_conditions(self, data):
        try:
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            
            # Рассчитываем индикаторы
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            
            # Получаем последние значения
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # Проверяем тренд по нескольким свечам
            last_3_closes = df['close'].tail(3)
            price_trend = (last_3_closes.iloc[-1] > last_3_closes.iloc[-2]) and (last_3_closes.mean() > df['close'].tail(6).mean())
            
            # Проверяем объем
            avg_volume = df['volume'].rolling(window=5).mean()
            volume_ok = float(last_candle['volume']) > float(avg_volume.iloc[-1] * 0.5)
            
            # Проверяем SMA и RSI
            above_sma = float(last_candle['close']) > float(last_candle['sma_20'])
            rsi_condition = config.RSI_SETTINGS['oversold'] < float(last_candle['rsi']) < config.RSI_SETTINGS['overbought']
            
            # Логируем условия...
            
            # Сигнал генерируется если:
            # 1. Есть растущий тренд
            # 2. Объем выше 50% от среднего
            # 3. Цена выше SMA20
            # 4. RSI в допустимом диапазоне
            signal = price_trend and volume_ok and above_sma and rsi_condition
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Ошибка проверки условий сигнала: {str(e)}")
            return False

    def _calculate_rsi(self, prices, period=14):
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
