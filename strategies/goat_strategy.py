import os
import sys
from pathlib import Path
import logging

# Добавляем корневую директорию проекта в PYTHONPATH
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from strategies.base_strategy import SignalStrategy
import numpy as np
import pandas as pd
import ta


class GoatStrategy(SignalStrategy):
    """
    GOAT (Greatest Of All Time) торговая стратегия
    Использует комбинацию технических индикаторов для определения точек входа и выхода
    """

    def __init__(self, config):
        """
        Инициализация параметров стратегии
        """
        self.config = config
        # Базовые параметры
        self.rsi_period = 2          # Сверхкороткий период RSI
        self.rsi_overbought = 70     # Стандартные уровни RSI
        self.rsi_oversold = 30
        self.ema_short = 2           # Сверхбыстрые EMA
        self.ema_medium = 3
        self.ema_long = 4

        # Дополнительные параметры
        self.atr_period = 3          # Минимальный период ATR
        self.volume_ma_period = 3    # Минимальный период объема
        self.trailing_stop_atr = 0.5 # Минимальный стоп
        self.min_volatility = 0.001  # Минимальная волатильность
        self.profit_target = 1.5     # Увеличенный профит

        # Веса для условий
        self.weights = {
            'rsi': 0.8,       # Максимальный вес на RSI
            'trend': 0.2,     # Минимальный вес на тренд
            'volume': 0.0,    # Игнорируем объем
            'volatility': 0.0 # Игнорируем волатильность
        }

        self.trades = []
        self.data = None

    def get_name(self):
        """Получение названия стратегии"""
        return "GOAT Strategy"

    def get_description(self):
        """Получение описания стратегии"""
        return """GOAT стратегия использует комбинацию технических индикаторов:
        - RSI для определения перекупленности/перепроданности
        - Три EMA для определения тренда
        - ATR для расчета волатильности и стоп-лоссов
        - Объемный анализ для подтверждения сигналов"""

    def get_parameters(self):
        """Получение параметров стратегии"""
        return {
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'ema_short': self.ema_short,
            'ema_medium': self.ema_medium,
            'ema_long': self.ema_long,
            'atr_period': self.atr_period,
            'volume_ma_period': self.volume_ma_period,
            'trailing_stop_atr': self.trailing_stop_atr
        }

    def validate_data(self, data):
        """Проверка наличия необходимых данных"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)

    def calculate_indicators(self, data):
        """Расчет необходимых индикаторов"""
        indicators = {}

        # RSI
        indicators['rsi'] = ta.momentum.RSIIndicator(data['close'], window=self.rsi_period).rsi()

        # EMA
        indicators['ema_short'] = ta.trend.EMAIndicator(data['close'], window=self.ema_short).ema_indicator()
        indicators['ema_medium'] = ta.trend.EMAIndicator(data['close'], window=self.ema_medium).ema_indicator()
        indicators['ema_long'] = ta.trend.EMAIndicator(data['close'], window=self.ema_long).ema_indicator()

        # ATR
        indicators['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'],
                                                           window=self.atr_period).average_true_range()

        # Volume MA
        indicators['volume_ma'] = ta.trend.SMAIndicator(data['volume'], window=self.volume_ma_period).sma_indicator()

        return indicators

    def calculate_signal_strength(self, conditions):
        """Расчет   илы сигнала на основе весов"""
        strength = 0
        for key, value in conditions.items():
            if key in self.weights:
                strength += self.weights[key] * (1 if value else 0)
        return strength

    def calculate_volatility(self, data):
        """Расчет волатильности"""
        try:
            # Используем последние 24 свечи для расчета
            returns = np.log(data['close'] / data['close'].shift(1))
            volatility = float(returns.tail(24).std() * np.sqrt(24) * 100)
            return volatility
        except Exception as e:
            logging.error(f"Ошибка расчета волатильности: {str(e)}")
            return 0.0

    def calculate_trend_strength(self, data):
        """Расчет силы тренда"""
        try:
            # Используем SMA для определения тренда
            sma20 = ta.trend.SMAIndicator(data['close'], window=20).sma_indicator()
            sma50 = ta.trend.SMAIndicator(data['close'], window=50).sma_indicator()

            # Рассчитываем процентную разницу между SMA
            trend_strength = ((sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]) * 100
            return float(trend_strength)
        except Exception as e:
            logging.error(f"Ошибка расчета силы тренда: {str(e)}")
            return 0.0

    def calculate_volume_ratio(self, data):
        """Расчет отношения текущего объема к среднему"""
        try:
            # Средний объем за последние 24 периода
            avg_volume = data['volume'].tail(24).mean()
            current_volume = data['volume'].iloc[-1]

            if avg_volume == 0:
                return 0.0

            volume_ratio = current_volume / avg_volume
            return float(volume_ratio)
        except Exception as e:
            logging.error(f"Ошибка расчета отношения объема: {str(e)}")
            return 0.0

    def generate_signal(self, data):
        """Генерация сигналов с упрощенной логикой"""
        try:
            # Получаем значения индикаторов
            indicators = self.calculate_indicators(data)
            rsi = indicators['rsi'].iloc[-1]
            ema_short = indicators['ema_short'].iloc[-1]
            ema_medium = indicators['ema_medium'].iloc[-1]
            
            # Проверяем базовые условия
            if len(self.trades) > 0:
                last_trade_time = self.trades[-1].get('timestamp')
                if last_trade_time and (data.index[-1] - last_trade_time).seconds < 1:
                    return "HOLD"
            
            # Упрощенная проверка объема
            volume_ok = True  # Игнорируем проверку объема
            
            # Сигнал на покупку (упрощенные условия)
            if (
                (rsi <= 30 or                  # Основное условие RSI
                (rsi <= 40 and ema_short > ema_medium))  # Дополнительное условие тренда
            ):
                return "BUY"
                
            # Сигнал на продажу (упрощенные условия)
            elif (
                (rsi >= 70 or                  # Основное условие RSI
                (rsi >= 60 and ema_short < ema_medium))  # Дополнительное условие тренда
            ):
                return "SELL"
                
            return "HOLD"
            
        except Exception as e:
            logging.error(f"Ошибка при генерации сигнала: {str(e)}")
            return "HOLD"

    def calculate_position_size(self, data, indicators):
        """Расчет размера позиции с агрессивным риском"""
        try:
            atr = indicators['atr'].iloc[-1]
            current_price = data['close'].iloc[-1]
            account_balance = self.config.get('account_balance', 10000)
            risk_per_trade = self.config.get('risk_per_trade', 0.05)  # Максимальный риск
            
            # Минимальный стоп-лосс
            stop_loss_atr = atr * 0.5
            stop_loss_price = current_price - stop_loss_atr
            
            # Расчет размера позиции
            risk_amount = account_balance * risk_per_trade
            position_size = risk_amount / (current_price - stop_loss_price)
            
            # Ограничение размера позиции
            max_position_size = account_balance * 0.4  # Максимальный размер
            position_size = min(position_size, max_position_size)
            
            return round(position_size, 2)
            
        except Exception as e:
            logging.error(f"Ошибка расчета размера позиции: {str(e)}")
            return 0.1

    def calculate_stop_loss(self, data, indicators, side):
        """Улучшенный расчет стоп-лосса"""
        current_price = data['close'].iloc[-1]
        atr = indicators['atr'].iloc[-1]

        if side == 'BUY':
            # Динамический стоп в зависимости от волатильности
            stop_distance = max(atr * self.trailing_stop_atr,
                                current_price * 0.01)  # Минимум 1%
            return current_price - stop_distance
        else:
            stop_distance = max(atr * self.trailing_stop_atr,
                                current_price * 0.01)
            return current_price + stop_distance

    def calculate_take_profit(self, data, indicators, side):
        """Расчет тейк-профита на основе ATR"""
        current_price = data['close'].iloc[-1]
        atr = indicators['atr'].iloc[-1]

        if side == 'BUY':
            return current_price + (atr * self.profit_target)
        else:
            return current_price - (atr * self.profit_target)

    def calculate_strategy_performance(self, signals, data):
        """Расчет эффективности стратегии"""
        wins = 0
        losses = 0
        total_profit_pct = 0
        position = None
        entry_price = 0

        for i, signal in enumerate(signals):
            if i >= len(data):
                break

            current_price = data['close'].iloc[i]

            if signal == 'BUY' and position is None:
                position = 'LONG'
                entry_price = current_price
            elif signal == 'SELL' and position == 'LONG':
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                total_profit_pct += profit_pct

                if profit_pct > 0:
                    wins += 1
                else:
                    losses += 1

                position = None

        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        return {
            'win_rate': win_rate,
            'total_profit_pct': total_profit_pct,
            'total_trades': wins + losses,
            'wins': wins,
            'losses': losses,
            'avg_profit_per_trade': total_profit_pct / (wins + losses) if (wins + losses) > 0 else 0
        }

    def calculate_trailing_stop(self, data, indicators, entry_price, side='BUY'):
        """
        Расчет трейлинг-стопа
        """
        atr = indicators['atr'].iloc[-1]
        current_price = data['close'].iloc[-1]

        if side == 'BUY':
            # Для длинной позиции
            stop_distance = atr * self.trailing_stop_atr
            trailing_stop = max(current_price - stop_distance,
                                entry_price - (stop_distance * 0.5))  # Минимальный стоп в половину ATR от входа
            return trailing_stop
        else:
            # Для короткой позиции
            stop_distance = atr * self.trailing_stop_atr
            trailing_stop = min(current_price + stop_distance,
                                entry_price + (stop_distance * 0.5))  # Минимальный стоп в половину ATR от входа
            return trailing_stop

    def calculate_rsi(self, data, period=14):
        """Расчет RSI"""
        try:
            rsi = ta.momentum.RSIIndicator(data['close'], window=period).rsi()
            return rsi.iloc[-1]
        except Exception as e:
            logging.error(f"Ошибка расчета RSI: {str(e)}")
            return 50  # Нейтральное значение при ошибке

    def add_trade(self, signal, timestamp):
        """Добавление информации о сделке"""
        if self.data is not None and not self.data.empty:
            trade = {
                'signal': signal,
                'timestamp': timestamp,
                'price': self.data['close'].iloc[-1]
            }
            self.trades.append(trade)

            # Оставляем только последние 100 сделок
            if len(self.trades) > 100:
                self.trades = self.trades[-100:]

    def load_data(self, data):
        """Загрузка данных для анализа"""
        self.data = data
        self.indicators = self.calculate_indicators(data)

    def _calculate_trend_score(self, data, trend_up, trend_down, momentum, price_trend):
        """Расчет оценки тренда"""
        try:
            score = 0

            # Оценка направления тренда
            if trend_up:
                score += 0.3
            elif trend_down:
                score -= 0.3

            # Оценка моментума
            score += min(max(momentum / 10, -0.3), 0.3)

            # Оценка тренда цены
            score += min(max(price_trend / 20, -0.2), 0.2)

            # Оценка консистентности движения
            price_changes = data['close'].pct_change().tail(5)
            consistent_moves = sum(1 for x in price_changes if (x > 0) == trend_up)
            score += (consistent_moves - 2.5) * 0.04

            return min(max(score, -1), 1)

        except Exception as e:
            logging.error(f"Ошибка расчета оценки тренда: {str(e)}")
            return 0

    def _calculate_volume_score(self, volume_ratio, volume_trend):
        """Расчет оценки объема"""
        try:
            score = 0

            # Оценка текущего объема
            score += min(max((volume_ratio - 0.7) / 2, 0), 0.4)

            # Оценка тренда объема
            score += min(max((volume_trend - 0.8) / 2, 0), 0.4)

            # Добавляем бонус за очень высокий объем
            if volume_ratio > 2:
                score += 0.2

            return min(max(score, 0), 1)

        except Exception as e:
            logging.error(f"Ошибка расчета оценки объема: {str(e)}")
            return 0

    def _analyze_liquidity(self, data):
        """Анализ ликвидности рынка"""
        try:
            # Рассчитываем спред между максимумами и минимумами
            high_low_spread = (data['high'] - data['low']) / data['low'] * 100
            avg_spread = high_low_spread.rolling(20).mean().iloc[-1]

            # Анализируем объем и его стабильность
            volume_std = data['volume'].rolling(20).std() / data['volume'].rolling(20).mean()
            volume_stability = volume_std.iloc[-1]

            # Оцениваем глубину рынка через объем
            depth_score = min(data['volume'].iloc[-1] / data['volume'].rolling(100).mean().iloc[-1], 2)

            # Смягчаем словия ликвидности
            return {
                'spread': avg_spread,
                'volume_stability': volume_stability,
                'depth_score': depth_score,
                'is_liquid': avg_spread < 2.5 and volume_stability < 0.8 and depth_score > 0.5  # Более мягкие условия
            }
        except Exception as e:
            logging.error(f"Ошибка анализа ликвидности: {str(e)}")
            return {'is_liquid': True}  # По умол��анию считаем рынок ликвидным

    def _analyze_market_conditions(self, data, rsi, trend_up, trend_down):
        """Расширенный анализ рыночны условий"""
        try:
            # Анализ волатильности
            volatility = self.calculate_volatility(data)
            volatility_ma = self.calculate_volatility(data.iloc[-20:])
            volatility_trend = 'rising' if volatility > volatility_ma else 'falling'

            # Анализ объемов
            volume_sma = data['volume'].rolling(20).mean()
            volume_trend = 'rising' if data['volume'].iloc[-1] > volume_sma.iloc[
                -1] * 0.8 else 'falling'  # Снижаем требование

            # Анализ моментума
            momentum = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5] * 100
            momentum_state = 'strong' if abs(momentum) > 2 else 'weak'  # Снижаем порог

            # Смягчаем условия торговли
            market_conditions = {
                'volatility_state': volatility_trend,
                'volume_state': volume_trend,
                'momentum_state': momentum_state,
                'is_tradeable': (
                        volatility > 0.5 and  # Снижаем минимальную волатильность
                        volatility < 8.0 and  # Увеличиваем максимальную в  латильно  ть
                        abs(momentum) < 15.0  # Увеличиваем допустимый моментум
                )
            }

            return market_conditions
        except Exception as e:
            logging.error(f"Ошибка анализа рыночных условий: {str(e)}")
            return {'is_tradeable': True}  # По умолчанию считаем рынок торгуемым

    def _calculate_market_correlation(self, data):
        """Расчет корреляции с общим рыночным трендом"""
        try:
            # Используем BTC как прокси для общего рынка
            market_data = data['close'].pct_change()

            # Рассчитываем корреляцию за последние 20 периодов
            correlation = market_data.rolling(20).corr(market_data)
            current_correlation = correlation.iloc[-1]

            # Определяем силу корреляции
            correlation_strength = abs(current_correlation)
            correlation_direction = 1 if current_correlation > 0 else -1

            return {
                'strength': correlation_strength,
                'direction': correlation_direction,
                'is_strong': correlation_strength > 0.7
            }
        except Exception as e:
            logging.error(f"Ошибка расчета корреляции: {str(e)}")
            return {'strength': 0, 'direction': 0, 'is_strong': False}

    def _analyze_impulse(self, data):
        """Анализ импульса движения"""
        try:
            # Рассчитываем изменение цены за разные периоды
            changes = {
                'short': data['close'].pct_change(3).iloc[-1] * 100,  # 3 перода
                'medium': data['close'].pct_change(7).iloc[-1] * 100,  # 7 периодов
                'long': data['close'].pct_change(14).iloc[-1] * 100  # 14 периодов
            }

            # Анализируем силу импульса
            impulse_strength = sum(1 for x in changes.values() if abs(x) > 1)
            impulse_direction = sum(1 for x in changes.values() if x > 0)

            return {
                'strength': impulse_strength / 3,  # Нормализуем от 0 до 1
                'direction': 1 if impulse_direction >= 2 else -1,
                'is_strong': impulse_strength >= 2
            }
        except Exception as e:
            logging.error(f"Ошибка анализа импульса: {str(e)}")
            return {'strength': 0, 'direction': 0, 'is_strong': False}

    def _check_signal_conditions(self, data, last_signal=None):
        """Проверка дополнительных условий для сигналов"""
        try:
            # Проверка последовательных сигналов
            if last_signal and self.trades:
                consecutive_count = 0
                for trade in reversed(self.trades):
                    if trade['signal'] != last_signal:
                        break
                    consecutive_count += 1

                if consecutive_count >= 2:  # Максимум 2 последовательных сигнала
                    logging.info(f"Слишком много последовательных сигналов {last_signal}: {consecutive_count}")
                    return False

            # Проверка волатильности
            recent_volatility = self.calculate_volatility(data.tail(5))
            avg_volatility = self.calculate_volatility(data.tail(20))

            if recent_volatility > avg_volatility * 1.5:  # Снижаем порог
                logging.info(f"Слишком высокая волатильность: {recent_volatility:.2f}% vs {avg_volatility:.2f}%")
                return False

            # Проверка объема
            recent_volume = data['volume'].tail(5).mean()
            avg_volume = data['volume'].tail(20).mean()

            if recent_volume < avg_volume * 0.7:  # Повышаем минимальный порог
                logging.info(f"Слишком низкий объем: {recent_volume:.2f} vs {avg_volume:.2f}")
                return False

            # Проверка RSI
            rsi = self.calculate_rsi(data)
            if rsi > 70 or rsi < 30:  # Добавляем проверку RSI
                logging.info(f"Экстремальное значение RSI: {rsi:.2f}")
                return False

            return True

        except Exception as e:
            logging.error(f"Ошибка проверки условий сигнала: {str(e)}")
            return False


# Добавляем код для тестирования сратегии при прямом запуске файла
if __name__ == '__main__':
    import yfinance as yf

    # Загружаем тестовые данные
    symbol = 'BTC-USD'
    data = yf.download(symbol, start='2023-01-01', end='2024-01-01', interval='1d')

    # Преобразуем MultiIndex в обычные колонки и приводим к нижнему регистру
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].lower() for col in data.columns]
    else:
        data.columns = data.columns.str.lower()

    # Инициализируем стратегию
    config = {
        'account_balance': 10000,
        'risk_per_trade': 0.02
    }
    strategy = GoatStrategy(config)

    # Рассчитываем индикаторы
    indicators = strategy.calculate_indicators(data)

    # Генерируем сигналы
    signals = []
    for i in range(len(data)):
        if i < 50:  # Пропускаем первые дни для накопления данных индикаторов
            continue

        current_data = data.iloc[:i + 1]
        current_indicators = {
            'rsi': indicators['rsi'].iloc[:i + 1],
            'ema_short': indicators['ema_short'].iloc[:i + 1],
            'ema_medium': indicators['ema_medium'].iloc[:i + 1],
            'ema_long': indicators['ema_long'].iloc[:i + 1],
            'atr': indicators['atr'].iloc[:i + 1],
            'volume_ma': indicators['volume_ma'].iloc[:i + 1]
        }

        signal = strategy.generate_signal(current_data)
        signals.append(signal)

        if signal != 'HOLD':
            print(f"Date: {data.index[i]}, Signal: {signal}, Price: {data['close'][i]:.2f}")
            if signal == 'BUY':
                stop_loss = strategy.calculate_stop_loss(current_data, current_indicators, 'BUY')
                take_profit = strategy.calculate_take_profit(current_data, current_indicators, 'BUY')
                position_size = strategy.calculate_position_size(current_data, current_indicators)
                print(f"Position Size: {position_size:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
            print("---")

        if i % 20 == 0:  # Каждые 20 вечей
            print(f"\nТекущие значения индикаторов (дата: {data.index[i]}):")
            print(f"RSI: {indicators['rsi'].iloc[i]:.2f}")
            print(f"EMA short: {indicators['ema_short'].iloc[i]:.2f}")
            print(f"EMA medium: {indicators['ema_medium'].iloc[i]:.2f}")
            print(f"EMA long: {indicators['ema_long'].iloc[i]:.2f}")
            print(f"ATR: {indicators['atr'].iloc[i]:.2f}")
            print(f"Volume MA: {indicators['volume_ma'].iloc[i]:.2f}")
            print("---")

    # Выодим стаистику сигналов
    total_signals = len([s for s in signals if s != 'HOLD'])
    buy_signals = len([s for s in signals if s == 'BUY'])
    sell_signals = len([s for s in signals if s == 'SELL'])

    print("\nСтатистика сигнало:")
    print(f"Всего сигналов: {total_signals}")
    print(f"Сигналов на покупку: {buy_signals}")
    print(f"Сигналов на продажу: {sell_signals}")