import numpy as np
from typing import Dict, List, Optional
import pandas as pd
import logging
import config
from binance.client import Client


class GoatStrategy:
    def __init__(self, risk_reward_ratio: float = 2.0, max_loss: float = 0.02, balance: float = None):
        self.risk_reward_ratio = risk_reward_ratio
        self.max_loss = max_loss
        self.balance = balance
        self.symbols = []
        self.positions = {}
        self.asset_metrics = {}
        self.logger = logging.getLogger('TradingBot')
        
        # Добавляем базовые параметры стратегии
        self.params = {
            'timeframe': '1m',
            'rsi_period': 14,
            'ema_periods': [9, 21],
            'volume_ma_period': 5,
            'min_volume_ratio': 0.8,
            'min_volume': 1000000,  # Минимальный объем в USDT
            'min_volatility': 0.5,  # Минимальная волатильность в %
            'max_volatility': 50,   # Максимальная волатильность в %
        }
        
        # Добавляем имя стратегии
        self.name = 'GOAT (Greatest Of All Time)'

    def analyze_market(self, client: Client) -> List[str]:
        """Анализирует рынок и возвращает список подходящих торговых пар"""
        try:
            self.logger.info("Начало анализа рынка...")
            
            # Получаем информацию о всех USDT парах
            tickers = client.get_ticker()
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
            
            suitable_pairs = []
            for pair in usdt_pairs:
                try:
                    symbol = pair['symbol']
                    
                    # Получаем статистику за 24 часа
                    volume_24h = float(pair['quoteVolume'])  # Объем в USDT
                    price_change = abs(float(pair['priceChangePercent']))  # Волатильность
                    
                    # Проверяем соответствие критериям
                    if (volume_24h >= self.params['min_volume'] and 
                        self.params['min_volatility'] <= price_change <= self.params['max_volatility']):
                        
                        # Рассчитываем метрики для оценки актива
                        metrics = {
                            'symbol': symbol,
                            'volume': volume_24h,
                            'volatility': price_change,
                            'trend': self._calculate_trend_score(client, symbol),
                        }
                        
                        # Рассчитываем общий скор
                        metrics['total_score'] = self._calculate_total_score(metrics)
                        
                        self.asset_metrics[symbol] = metrics
                        suitable_pairs.append(symbol)
                        
                        self.logger.info(
                            f"Актив {symbol}: "
                            f"Волатильность={price_change:.2f}%, "
                            f"Объем=${volume_24h/1000000:.1f}M, "
                            f"Тренд={metrics['trend']:.2f}, "
                            f"Общий скор={metrics['total_score']:.2f}"
                        )
                        
                except Exception as e:
                    self.logger.error(f"Ошибка анализа пары {symbol}: {str(e)}")
                    continue
            
            # Сортируем пары по общему скору и берем топ-5
            sorted_pairs = sorted(
                suitable_pairs,
                key=lambda x: self.asset_metrics[x]['total_score'],
                reverse=True
            )[:5]
            
            self.logger.info(f"Найдено {len(suitable_pairs)} подходящих пар")
            self.logger.info(f"Топ-5 пар по скору:")
            for symbol in sorted_pairs:
                metrics = self.asset_metrics[symbol]
                self.logger.info(
                    f"{symbol}: "
                    f"Скор={metrics['total_score']:.2f}, "
                    f"Объем=${metrics['volume']/1000000:.1f}M, "
                    f"Волатильность={metrics['volatility']:.1f}%"
                )
            
            return sorted_pairs
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа рынка: {str(e)}")
            return []

    def _calculate_trend_score(self, client: Client, symbol: str) -> float:
        """Рассчитывает оценку тренда"""
        try:
            # Получаем последние 100 свечей
            klines = client.get_klines(
                symbol=symbol,
                interval=self.params['timeframe'],
                limit=100
            )
            
            closes = pd.Series([float(k[4]) for k in klines])
            
            # Рассчитываем EMA
            ema9 = closes.ewm(span=9).mean()
            ema21 = closes.ewm(span=21).mean()
            
            # Если короткая EMA выше длинной - восходящий тренд
            trend_score = 1 if ema9.iloc[-1] > ema21.iloc[-1] else 0
            
            # Добавляем силу тренда
            trend_strength = abs((ema9.iloc[-1] - ema21.iloc[-1]) / ema21.iloc[-1])
            
            return trend_score * (1 + trend_strength)
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета тренда для {symbol}: {str(e)}")
            return 0

    def _calculate_total_score(self, metrics: Dict) -> float:
        """Рассчитывает общий скор актива"""
        try:
            # Веса для разных метрик
            weights = {
                'volume': 0.3,
                'volatility': 0.3,
                'trend': 0.4
            }
            
            # Нормализуем метрики
            volume_score = min(metrics['volume'] / (self.params['min_volume'] * 10), 1)
            volatility_score = min(metrics['volatility'] / self.params['max_volatility'], 1)
            trend_score = metrics['trend']
            
            # Рассчитываем взвешенный скор
            total_score = (
                volume_score * weights['volume'] +
                volatility_score * weights['volatility'] +
                trend_score * weights['trend']
            )
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета общего скора: {str(e)}")
            return 0

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Рассчитывает Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета ATR: {str(e)}")
            return pd.Series(index=data.index)

    def update_trading_pairs(self, pairs):
        """Обновляет список торгуемых пар"""
        self.symbols = pairs
        self._update_allocation()
        
    def _calculate_rsi(self, prices, period=None):
        """Рассчитывает RSI"""
        if period is None:
            period = self.params['rsi_period']
            
        # Используем ta-lib для расчета RSI
        try:
            import talib
            return talib.RSI(prices, timeperiod=period)
        except ImportError:
            # Альтернативный расчет без ta-lib
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

    def analyze_signal(self, data):
        """Анализирует данные и возвращает торговый сигнал"""
        try:
            # Проверяем наличие необходимых данных
            if data is None or len(data) < self.params['rsi_period']:
                return None
                
            # Рассчитываем индикаторы
            data['EMA9'] = data['close'].ewm(span=self.params['ema_periods'][0]).mean()
            data['EMA21'] = data['close'].ewm(span=self.params['ema_periods'][1]).mean()
            data['RSI'] = self._calculate_rsi(data['close'])
            data['ATR'] = self._calculate_atr(data)
            
            return self._check_signal_conditions(data)
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа сигнала: {str(e)}")
            return None

    def _check_signal_conditions(self, data):
        """Проверяет условия для генерации сигнала"""
        try:
            # Получаем последние значения
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            ema9 = data['EMA9'].iloc[-1]
            ema21 = data['EMA21'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(5).mean().iloc[-1]
            
            # Проверяем условия
            trend_ok = ema9 > ema21
            momentum_ok = current_price > prev_price
            volume_ok = volume > (avg_volume * 0.8)
            rsi_ok = 30 <= rsi <= 70
            
            if all([trend_ok, momentum_ok, volume_ok, rsi_ok]):
                return 'BUY'
            elif all([not trend_ok, not momentum_ok, volume_ok, rsi_ok]):
                return 'SELL'
                
            return 'HOLD'
            
        except Exception as e:
            self.logger.error(f"Ошибка проверки условий сигнала: {str(e)}")
            return 'HOLD'

    def _update_allocation(self):
        """Обновляет распределение средств между парами"""
        try:
            if not self.symbols:
                return
            
            # Равномерное распределение
            allocation_per_pair = 1.0 / len(self.symbols)
            
            self.allocations = {
                symbol: allocation_per_pair 
                for symbol in self.symbols
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления аллокации: {str(e)}")

    def analyze(self, market_data):
        """Анализирует рыночные данные и возвращает торговые сигналы"""
        try:
            signals = []
            
            for symbol in self.symbols:
                try:
                    if symbol not in market_data:
                        self.logger.warning(f"Нет данных для {symbol}")
                        continue
                    
                    # Преобразуем данные в DataFrame
                    data = pd.DataFrame(market_data[symbol])
                    if len(data) < 2:
                        self.logger.warning(f"Недостаточно данных для {symbol}")
                        continue
                    
                    # Рассчитываем индикаторы
                    data['EMA9'] = data['close'].ewm(span=self.params['ema_periods'][0]).mean()
                    data['EMA21'] = data['close'].ewm(span=self.params['ema_periods'][1]).mean()
                    data['RSI'] = self._calculate_rsi(data['close'])
                    data['ATR'] = self._calculate_atr(data)
                    
                    # Получаем последние значения
                    current_price = float(data['close'].iloc[-1])
                    prev_price = float(data['close'].iloc[-2])
                    ema9 = float(data['EMA9'].iloc[-1])
                    ema21 = float(data['EMA21'].iloc[-1])
                    rsi = float(data['RSI'].iloc[-1])
                    volume = float(data['volume'].iloc[-1])
                    avg_volume = float(data['volume'].rolling(5).mean().iloc[-1])
                    
                    self.logger.info(f"\n{'='*50}")
                    self.logger.info(f"Анализ пары: {symbol}")
                    self.logger.info(f"Текущая цена: {current_price:.8f}")
                    self.logger.info(f"Предыдущая цена: {prev_price:.8f}")
                    self.logger.info(f"EMA9: {ema9:.8f}")
                    self.logger.info(f"EMA21: {ema21:.8f}")
                    self.logger.info(f"RSI: {rsi:.2f}")
                    self.logger.info(f"Объем: {volume:.2f}")
                    self.logger.info(f"Средний объем: {avg_volume:.2f}")
                    
                    # Проверяем условия
                    trend_ok = ema9 > ema21
                    momentum_ok = current_price > prev_price
                    volume_ok = volume > (avg_volume * self.params['min_volume_ratio'])
                    rsi_ok = 30 <= rsi <= 70
                    
                    self.logger.info("\nПроверка условий:")
                    self.logger.info(f"- Тренд (EMA9 > EMA21): {trend_ok}")
                    self.logger.info(f"- Моментум (цена растет): {momentum_ok}")
                    self.logger.info(f"- Объем (> {self.params['min_volume_ratio']*100}% от среднего): {volume_ok}")
                    self.logger.info(f"- RSI (30-70): {rsi_ok}")
                    
                    # Определяем сигнал
                    signal = None
                    if all([trend_ok, momentum_ok, volume_ok, rsi_ok]):
                        signal = 'BUY'
                    elif all([not trend_ok, not momentum_ok, volume_ok, rsi_ok]):
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'
                        reasons = []
                        if not trend_ok:
                            reasons.append("тренд не подтвержден")
                        if not momentum_ok:
                            reasons.append("нет роста цены")
                        if not volume_ok:
                            reasons.append("недостаточный объем")
                        if not rsi_ok:
                            reasons.append("RSI вне диапазона")
                        self.logger.info(f"Сигнал HOLD. Причины: {', '.join(reasons)}")
                    
                    self.logger.info(f"\nИтоговый сигнал: {signal}")
                    
                    if signal not in ['HOLD', None]:
                        # Рассчитываем параметры позиции
                        atr = float(data['ATR'].iloc[-1])
                        
                        if signal == 'BUY':
                            stop_loss = current_price - (2 * atr)
                            take_profit = current_price + (4 * atr)
                        else:  # SELL
                            stop_loss = current_price + (2 * atr)
                            take_profit = current_price - (4 * atr)
                        
                        stop_loss_percent = abs((stop_loss - current_price) / current_price * 100)
                        take_profit_percent = abs((take_profit - current_price) / current_price * 100)
                        
                        signal_info = {
                            'symbol': symbol,
                            'direction': signal,
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'stop_loss_percent': stop_loss_percent,
                            'take_profit_percent': take_profit_percent
                        }
                        
                        self.logger.info("\nПараметры сигнала:")
                        self.logger.info(f"- Вход: {current_price:.8f}")
                        self.logger.info(f"- Стоп: {stop_loss:.8f} ({stop_loss_percent:.2f}%)")
                        self.logger.info(f"- Цель: {take_profit:.8f} ({take_profit_percent:.2f}%)")
                        
                        signals.append(signal_info)
                
                except Exception as e:
                    self.logger.error(f"Ошибка анализа {symbol}: {str(e)}")
                    continue
                
            self.logger.info(f"\nВсего найдено сигналов: {len(signals)}")
            return signals
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа рынка: {str(e)}")
            return []
