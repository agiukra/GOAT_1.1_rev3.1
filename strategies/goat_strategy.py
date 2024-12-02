import numpy as np
from typing import Dict, List, Optional
import pandas as pd
import logging
import config


class GOATStrategy:
    def __init__(self, risk_reward_ratio: float, max_loss: float):
        self.risk_reward_ratio = risk_reward_ratio
        self.max_loss = max_loss
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
            
            # Проверяем условия
            price_trend = float(last_candle['close']) > float(prev_candle['close'])
            volume_increase = float(last_candle['volume']) > float(prev_candle['volume'])
            above_sma = float(last_candle['close']) > float(last_candle['sma_20'])
            rsi_condition = 30 < float(last_candle['rsi']) < 70
            
            # Логируем все условия с конкретными значениями
            self.logger.info(f"\nАнализ условий для сигнала:")
            self.logger.info(f"1. Тренд растущий: {price_trend}")
            self.logger.info(f"   - Текущая цена: {last_candle['close']:.8f}")
            self.logger.info(f"   - Предыдущая цена: {prev_candle['close']:.8f}")
            self.logger.info(f"   - Изменение: {((last_candle['close'] - prev_candle['close'])/prev_candle['close']*100):.2f}%")
            
            self.logger.info(f"2. Объем растет: {volume_increase}")
            self.logger.info(f"   - Текущий объем: {last_candle['volume']:.2f}")
            self.logger.info(f"   - Предыдущий объем: {prev_candle['volume']:.2f}")
            self.logger.info(f"   - Изменение: {((last_candle['volume'] - prev_candle['volume'])/prev_candle['volume']*100):.2f}%")
            
            self.logger.info(f"3. Цена выше SMA20: {above_sma}")
            self.logger.info(f"   - Цена: {last_candle['close']:.8f}")
            self.logger.info(f"   - SMA20: {last_candle['sma_20']:.8f}")
            self.logger.info(f"   - Разница: {((last_candle['close'] - last_candle['sma_20'])/last_candle['sma_20']*100):.2f}%")
            
            self.logger.info(f"4. RSI в зоне 30-70: {rsi_condition}")
            self.logger.info(f"   - RSI: {last_candle['rsi']:.2f}")
            
            # Сигнал генерируется только если все условия выполнены
            signal = price_trend and volume_increase and above_sma and rsi_condition
            self.logger.info(f"\nИтоговый сигнал: {'ЕСТЬ' if signal else 'НЕТ'}")
            if not signal:
                self.logger.info("Причины отсутствия сигнала:")
                if not price_trend:
                    self.logger.info("- Цена не растет")
                if not volume_increase:
                    self.logger.info("- Объем не растет")
                if not above_sma:
                    self.logger.info("- Цена ниже SMA20")
                if not rsi_condition:
                    self.logger.info("- RSI вне диапазона 30-70")
            
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
