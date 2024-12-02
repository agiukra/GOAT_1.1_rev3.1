import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from binance.client import Client
import config
import warnings
import time
import json
import logging
from binance.exceptions import BinanceAPIException
import logging.handlers
from technical_indicators import TechnicalIndicators
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
from strategies.strategy_manager import StrategyManager
from strategies.goat_strategy import GOATStrategy
import sys

warnings.filterwarnings('ignore')

# В начале файла, после импортов
def setup_logging():
    """Настройка системы логирования"""
    # Удаляем все существующие обработчики
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Отключаем propagation
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Форматтер для логов
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Консольный вывод
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # Файловый вывод
    file_handler = logging.handlers.RotatingFileHandler(
        'trading_bot.log',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Настраиваем корневой логгер для стратегии
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Добавляем те же обработчики к корневому логгеру
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return logger

class TradingBot:
    def __init__(self):
        """Инициализация торгового бота"""
        self.logger = setup_logging()
        self.data = None
        self.market_data = {}  # Словарь для хранения данных по каждому символу
        self.indicators = None
        self.trades = []
        self.last_signal = 'HOLD'
        self.balance = None
        self.signal_queue = Queue()
        self.data_lock = threading.Lock()

        try:
            os.makedirs('trades', exist_ok=True)
            self._initialize_bot_state()
            self.client = Client(config.API_KEY, config.API_SECRET)
            self.logger.info("API подключен успешно")
            self._update_balance()
        except Exception as e:
            self.logger.error(f"Ошибка инициализации бота: {str(e)}", exc_info=True)
            self.client = None

        # Инициализируем менеджер стратегий вместо одной стратегии
        self.strategy_manager = StrategyManager(config.INDICATOR_PARAMS)

    def _initialize_bot_state(self):
        """Инициализация начального состояния бота"""
        try:
            initial_state = {
                'status': 'initializing',
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'balance': None,
                'current_price': None,
                'last_signal': 'HOLD',
                'technical_indicators': {
                    'rsi': None,
                    'sma20': None,
                    'sma50': None,
                    'bb_high': None,
                    'bb_low': None,
                    'bb_mid': None,
                    'price_change_24h': None,
                    'volume_24h': None,
                    'volatility_24h': None,
                    'highest_24h': None,
                    'lowest_24h': None,
                    'trend': None,
                    'trend_strength': None,
                    'rsi_signal': None,
                    'bb_position': None
                },
                'trading_statistics': {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'last_trades': []
                },
                'open_positions': [],
                'risk_metrics': {
                    'max_position_size': None,
                    'max_daily_risk': None,
                    'current_risk_level': None
                }
            }

            with open('bot_state.json', 'w') as f:
                json.dump(initial_state, f, indent=4)

            self.logger.info("Инициализировано начальное состояние бота")

        except Exception as e:
            self.logger.error(f"Ошибка инициализации состояния бота: {str(e)}", exc_info=True)

    def generate_test_data(self):
        """Генерация тестовых данных если API недоступен"""
        print("Используютя тестове даные")
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        np.random.seed(42)
        prices = np.random.randn(100).cumsum() * 100 + 50000

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(100) * 10,
            'high': prices + np.random.randn(100) * 20,
            'low': prices - np.random.randn(100) * 20,
            'close': prices,
            'volume': np.random.uniform(1, 100, 100) * 1000
        })

        df.set_index('timestamp', inplace=True)
        self.data = df
        return df

    def fetch_data(self):
        """Получение реальных данных с Binance"""
        if not self.client:
            return self.generate_test_data()

        try:
            success = True
            for symbol in config.TRADING_PARAMS['symbols']:
                try:
                    print(f"Получение данных для {symbol}")
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=config.TRADING_PARAMS['interval'],
                        limit=config.TRADING_PARAMS['limit']
                    )

                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])

                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)

                    df.set_index('timestamp', inplace=True)
                    self.market_data[symbol] = df
                    
                except Exception as e:
                    self.logger.error(f"Ошибка получения данных для {symbol}: {str(e)}")
                    success = False

            return success

        except Exception as e:
            self.logger.error(f"Общая ошибка при получении данных: {str(e)}")
            return False

    def load_data(self, data):
        self.data = data
        self.indicators = TechnicalIndicators(data)
    
    def analyze_market(self):
        # Теперь используем кэшированные индикаторы
        sma_20 = self.indicators.calculate_sma(20)
        ema_50 = self.indicators.calculate_ema(50)
        rsi = self.indicators.calculate_rsi()
        
        # Остальная логика анализа...

    def generate_signal(self):
        """Генерация торгового сигнала с автоматическим вбром стратегии"""
        try:
            # Выбираем подходящую стратегию
            strategy = self.strategy_manager.select_strategy(self.data)
            
            # Генерируем сигнал используя выбранную стратегию
            return strategy.generate_signal(self.data, self.indicators)
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации сигнала: {str(e)}")
            return 'HOLD'

    def calculate_position_metrics(self, signal, symbol):
        """Расчет параметров позиции"""
        try:
            if symbol not in self.market_data:
                self.logger.error(f"Нет данных для символа {symbol}")
                return None
            
            market_data = self.market_data[symbol]
            if not market_data:
                self.logger.error(f"Пустые данные для символа {symbol}")
                return None
            
            entry_price = market_data[-1]['close']
            
            # Используем текущий баланс
            if self.balance is None:
                self.logger.error("Балас не определен")
                return None
            
            # Расчитываем риск от текущего баланса
            risk_percentage = config.TRADING_PARAMS.get('risk_percentage', 1)  # 1% по умолчанию
            risk_amount = self.balance * (risk_percentage / 100)

            if signal == 'SELL':
                stop_loss = entry_price * 1.01  # +1% от цены вода
                take_profit = entry_price * 0.97  # -3% от цены входа
            elif signal == 'BUY':
                stop_loss = entry_price * 0.99  # -1% от цены входа
                take_profit = entry_price * 1.03  # +3% от цены входа
            else:
                return None

            # Проверяем минимальный баланс для торговли
            min_balance = config.RISK_MANAGEMENT['risk_limits'].get('min_balance', 100)
            if self.balance < min_balance:
                self.logger.warning(f"Недостаточный баланс для торговли: {self.balance} USDT (минимум: {min_balance} USDT)")
                return None

            position_size = risk_amount / abs(entry_price - stop_loss)
            potential_profit = abs(take_profit - entry_price) * position_size

            # Проверяем минимальный размер позиции
            min_position_value = config.RISK_MANAGEMENT['position_sizing'].get('min_position_value', 50)
            position_value = position_size * entry_price
            if position_value < min_position_value:
                self.logger.warning(f"Слишком маленький размер позиции: {position_value:.2f} USDT (минимум: {min_position_value} USDT)")
                return None

            return {
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'potential_profit': potential_profit,
                'position_value': position_value
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при расчете параметров позиции: {str(e)}", exc_info=True)
            return None

    def save_trade(self, signal, metrics):
        """Сохранение информации о сделке"""
        trade_info = {
            'timestamp': datetime.now(),
            'signal': signal,
            **metrics
        }
        self.trades.append(trade_info)

        # Сохраняем в CSV
        df = pd.DataFrame([trade_info])
        df.to_csv(f'trades/trade_{len(self.trades)}.csv', index=False)

    def print_analysis(self, signal, metrics):
        """Вывод торгового анализа"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_row = self.data.iloc[-1]
        
        try:
            # Расчет базовых метрик
            price_change_24h = ((last_row['close'] - self.data['close'].iloc[-24]) / self.data['close'].iloc[-24] * 100)
            volume_24h = self.data['volume'].iloc[-24:].sum()
            
            # Анализ тренда
            trend_direction = "Восходящий" if last_row['trend_direction'] == 1 else "Нисходящий" if last_row['trend_direction'] == -1 else "Боковой"
            trend_strength = float(last_row['trend_strength'])
            
            # Анализ влатильности
            volatility = self._calculate_volatility()
            bb_position = float(last_row['bb_pct']) * 100
            
            # Расширенная рыночная статистика
            market_info = (
                f"\nРЫНОЧНЫЙ АНАЛИЗ | {current_time}\n"
                f"{'='*50}\n"
                f"\nЦЕНОВЫЕ ПОКАЗАТЕЛИ:\n"
                f"- Текущая цена: {last_row['close']:.2f} USDT\n"
                f"- Изменение (24ч): {price_change_24h:+.2f}%\n"
                f"- Волатильность: {volatility:.2f}%\n"
                f"- Объем (24ч): {volume_24h:.2f} USDT\n"
                f"\nТРЕНДОВЫЙ АНАЛИЗ:\n"
                f"- Направление: {trend_direction}\n"
                f"- Сила тренда: {trend_strength:.2f}%\n"
                f"- MA Cross: {'Бычье' if last_row['ma_cross'] == 1 else 'Медвежье' if last_row['ma_cross'] == -1 else 'Нет'}\n"
                f"\nТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:\n"
                f"- RSI ({config.INDICATOR_PARAMS['rsi']['period']}): {last_row['rsi']:.2f}\n"
                f"- RSI сигнал: {'Перепродан' if last_row['rsi_signal'] == 1 else 'Перекуплен' if last_row['rsi_signal'] == -1 else 'Нейтральный'}\n"
                f"- Положение BB: {bb_position:.1f}%\n"
                f"- Momentum: {last_row['momentum']:.2f}\n"
                f"\nУРОВНИ:\n"
                f"- Сопротивление: {self._calculate_resistance_level():.2f}\n"
                f"- Поддержка: {self._calculate_support_level():.2f}\n"
                f"\nОБЪЕМ:\n"
                f"- Тип: {self._analyze_volume()}\n"
                f"{'='*50}"
            )

            # Добавляем инфорацию об открытых позициях
            positions = self._get_open_positions()
            if positions:
                total_value = sum(pos['value_usdt'] for pos in positions)
                market_info += (
                    f"\nОТКРЫТЫЕ ПОЗИЦИИ (Общая стоимость: {total_value:.2f} USDT):\n"
                    f"{'='*50}\n"
                )
                for pos in positions:
                    market_info += (
                        f"- {pos['asset']:<8} | "
                        f"Количество: {pos['amount']:<12.8f} | "
                        f"Цена: {pos['current_price']:<10.2f} USDT | "
                        f"Стомость: {pos['value_usdt']:<10.2f} USDT "
                        f"({pos['value_usdt']/total_value*100:>5.1f}%)\n"
                    )

            # Добавляем статистику торговли
            stats = self._get_trading_statistics()
            if stats:
                market_info += (
                    f"\nСТАТИСТИКА ТРГВЛИ:\n"
                    f"{'='*50}\n"
                    f"- Всего сделок: {stats['total_trades']}\n"
                    f"- Успешнх сделок: {stats['successful_trades']}\n"
                    f"- еудачных сделок: {stats['failed_trades']}\n"
                    f"- Процент успешых: {stats['win_rate']:.2f}%\n"
                    f"- Общая прибыль: {stats['total_profit']:.2f} USDT\n"
                    f"\nПОСЛЕДНИЕ СДЕЛКИ:\n"
                )
                
                for trade in stats['recent_trades']:
                    profit = trade.get('profit', 0)
                    market_info += (
                        f"- {trade['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                        f"{trade['signal']} | "
                        f"Прибыль: {profit:+.2f} USDT\n"
                    )

            self.logger.info(market_info)
            
            # Если есть сигнал и метрики, выводим торговую информацию
            if signal != 'HOLD' and metrics:
                trade_info = self._format_trade_info(signal, metrics)
                self.logger.info(trade_info)
                
        except Exception as e:
            self.logger.error(f"Ошибка при формировании анализа: {str(e)}", exc_info=True)

    def execute_trade(self, signal, metrics):
        """Выполнение торговой операции"""
        if not self.client:
            self.logger.error("API не подключе. Тоговля невозможна.")
            return False

        try:
            # Получаем актуальнй баланс
            account = self.client.get_account()
            available_balance = float([asset for asset in account['balances']
                                       if asset['asset'] == 'USDT'][0]['free'])

            if available_balance < metrics['risk_amount']:
                self.logger.warning(
                    f"Недостаточ средст. Доступно: {available_balance} USDT, "
                    f"Тебуется: {metrics['risk_amount']} USDT"
                )
                return False

            side = Client.SIDE_BUY if signal == 'BUY' else Client.SIDE_SELL

            # Получаем информацию о символе для правильного округлени
            symbol_info = self.client.get_symbol_info(config.TRADING_PARAMS['symbol'])
            lot_size_filter = next(filter(lambda x: x['filterType'] == 'LOT_SIZE',
                                          symbol_info['filters']))

            # Округляем количество согласно требованиям биржи
            step_size = float(lot_size_filter['stepSize'])
            precision = len(str(step_size).split('.')[-1].rstrip('0'))
            quantity = round(metrics['position_size'], precision)

            # Соаем основной ордер
            order = self.client.create_order(
                symbol=config.TRADING_PARAMS['symbol'],
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )

            # Получаем цену исполнения
            executed_price = float(order['fills'][0]['price'])

            # Коррекируем стоп-лосс и тейк-профит оносительно цены сполнения
            if signal == 'BUY':
                stop_loss = executed_price * 0.99
                take_profit = executed_price * 1.03
            else:
                stop_loss = executed_price * 1.01
                take_profit = executed_price * 0.97

            # Создаем OCO ордер (комбинированный стоп-лосс и тейк-профит)
            oco_order = self.client.create_oco_order(
                symbol=config.TRADING_PARAMS['symbol'],
                side=Client.SIDE_SELL if signal == 'BUY' else Client.SIDE_BUY,
                quantity=quantity,
                stopPrice=str(stop_loss),
                stopLimitPrice=str(stop_loss),
                price=str(take_profit)
            )

            print(f"Рыночный ордр выполнен: {order['orderId']}")
            print(f"Установлен OCO ордер: {oco_order['orderReports'][0]['orderId']}")

            # Сохраняем детали сделки
            trade_details = {
                'order_id': order['orderId'],
                'executed_price': executed_price,
                'quantity': quantity,
                'side': side,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'oco_order_id': oco_order['orderReports'][0]['orderId']
            }

            self.save_trade_details(trade_details)
            return True

        except BinanceAPIException as e:
            if e.code == -2010:  # Insufficient balance
                self.logger.warning(f"Недостаточно средств для выполнения ордера: {e.message}")
            elif e.code == -1013:  # Invalid quantity
                self.logger.error(f"Неверное количество в ордере: {e.message}")
            elif e.code == -1021:  # Timestamp for this request is outside of the recvWindow
                self.logger.error("Ошибка инхронизации времени с сервером")
            else:
                self.logger.error(f"Ошибка API Binance: {e.code} - {e.message}")
            return False
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при выполнении ордера: {str(e)}", exc_info=True)
            return False

    def save_trade_details(self, trade_details):
        """Сохранение деталей сделки"""
        filename = f'trades/trade_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(trade_details, f, indent=4)

    def save_bot_state(self, state_info):
        """Сохранение состояния бота"""
        state = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_check': state_info,
            'is_running': True,
            'last_signal': state_info.get('signal'),
            'current_price': state_info.get('current_price'),
            'balance': state_info.get('balance')
        }

        with open('bot_state.json', 'w') as f:
            json.dump(state, f, indent=4)

    def update_bot_state(self):
        """Обновление состояния бота"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not self.market_data:
                raise ValueError("Отсутствуют рыночные данные")

            state = {
                'status': 'active',
                'last_update': current_time,
                'balance': float(self.balance) if self.balance is not None else 0,
                'symbols': {},
                'open_positions': self._get_open_positions() or [],
                'trading_stats': self._get_trading_statistics() or {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0
                }
            }

            # Добавляем информацию по каждому символу
            for symbol, data in self.market_data.items():
                if not data:
                    continue
                    
                last_candle = data[-1]
                state['symbols'][symbol] = {
                    'current_price': float(last_candle['close']),
                    'volume_24h': sum(d['volume'] for d in data[-24:]),
                    'price_change_24h': (
                        (last_candle['close'] - data[-24]['close']) / data[-24]['close'] * 100
                        if len(data) >= 24 else None
                    ),
                    'high_24h': max(d['high'] for d in data[-24:]),
                    'low_24h': min(d['low'] for d in data[-24:])
                }

            with open('bot_state.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4, ensure_ascii=False)

            self.logger.info(f"Состояние бота успешно обновлено: {current_time}")

        except ValueError as e:
            self.logger.warning(f"Ошибка валидации данных: {str(e)}")
            self._save_error_state(str(e))
        except Exception as e:
            self.logger.error("Критическая ошибка при обновлении состояния", exc_info=True)
            self._save_error_state(str(e))

    def _calculate_volatility(self):
        """Расчет волатильности"""
        try:
            # Используем последние 24 часа для расчета
            returns = np.log(self.data['close'] / self.data['close'].shift(1))
            return float(returns.tail(24).std() * np.sqrt(24) * 100)
        except Exception as e:
            self.logger.error(f"Ошибка расчета волатильности: {str(e)}", exc_info=True)
            return 0.0

    def _calculate_market_strength(self, last_row):
        """Расчет илы рынка"""
        try:
            # Рассчитываем силу тренда
            trend_strength = abs(
                (float(last_row['sma20']) - float(last_row['sma50'])) / 
                float(last_row['sma50']) * 100
            )
            
            # Рассчитываем моментум
            momentum = float(last_row['momentum']) if 'momentum' in last_row else 0
            
            # Рассчитываем отношение объема
            volume_ratio = (
                float(last_row['volume']) / 
                self.data['volume'].tail(24).mean()
            )
            
            # Определяем наденость тренда
            reliability = 'high' if (
                trend_strength > 2 and 
                abs(momentum) > 100 and 
                volume_ratio > 1.2
            ) else 'medium' if (
                trend_strength > 1 and 
                abs(momentum) > 50 and 
                volume_ratio > 0.8
            ) else 'low'

            return {
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volume_ratio': volume_ratio,
                'reliability': reliability
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета силы рынка: {str(e)}", exc_info=True)
            return {
                'trend_strength': 0,
                'momentum': 0,
                'volume_ratio': 1,
                'reliability': 'low'
            }

    def _determine_market_phase(self):
        """Определение фазы рынка"""
        try:
            last_row = self.data.iloc[-1]
            rsi = float(last_row['rsi'])
            price = float(last_row['close'])
            sma20 = float(last_row['sma20'])
            bb_high = float(last_row['bb_high'])
            bb_low = float(last_row['bb_low'])

            if price > bb_high and rsi > 70:
                return 'overbought'
            elif price < bb_low and rsi < 30:
                return 'oversold'
            elif price > sma20 and rsi > 50:
                return 'uptrend'
            elif price < sma20 and rsi < 50:
                return 'downtrend'
            else:
                return 'consolidation'
        except Exception as e:
            self.logger.error(f"Ошибка определения фазы рынка: {str(e)}", exc_info=True)
            return None

    def _calculate_risk_metrics(self, balance):
        """Расчет метрик риска"""
        try:
            if balance and hasattr(config, 'TRADING_PARAMS'):
                params = config.TRADING_PARAMS

                # Провряем наличие необходимых параметров
                max_position_size = params.get('max_position_size', 0.1)  # 10% от баланса по умолчанию
                max_daily_risk = params.get('max_daily_risk', 0.02)  # 2% от баланса по умолчанию
                risk_per_trade = params.get('risk_per_trade', 0.01)  # 1% от баланса по умолчанию

                return {
                    'max_position_size': float(balance * max_position_size),
                    'max_daily_risk': float(balance * max_daily_risk),
                    'current_risk_level': float(
                        len([t for t in self.trades if t.get('status') == 'open']) * balance * risk_per_trade),
                    'available_risk': float(balance * max_daily_risk - (
                                len([t for t in self.trades if t.get('status') == 'open']) * balance * risk_per_trade))
                }
            return {
                'max_position_size': None,
                'max_daily_risk': None,
                'current_risk_level': None,
                'available_risk': None
            }
        except Exception as e:
            self.logger.error(f"Ошибка расчета метрк риска: {str(e)}", exc_info=True)
            return {
                'max_position_size': None,
                'max_daily_risk': None,
                'current_risk_level': None,
                'available_risk': None
            }

    def run_auto_trading(self):
        """Запуск автоматической торговли"""
        try:
            print("\nЗапуск автоматической торговли...")
            
            # Инициализируем бота и стратегию
            self.initialize()
            
            cycle_count = 0

            while True:
                try:
                    cycle_count += 1
                    self.logger.info(f"\nЦикл #{cycle_count}")
                    
                    # Проверяем получение данных
                    self.logger.info("Получение данных...")
                    if not self.fetch_data():
                        self.logger.error("Не удалось получить данные")
                        continue
                    
                    self.logger.info(f"Получены данные для {len(self.market_data)} символов")
                    
                    # Анализируем рынок
                    self.logger.info("Анализ рынка...")
                    self.logger.info(f"Анализируемые символы: {', '.join(self.market_data.keys())}")
                    
                    signals = self.strategy.analyze(self.market_data)
                    
                    self.logger.info(f"Найдено сигналов: {len(signals)}")
                    
                    if signals:
                        self.process_signals(signals)
                    else:
                        self.logger.info("Нет торговых сигналов")
                    
                    time.sleep(config.TRADING_PARAMS.get('interval_seconds', 300))
                    
                except Exception as e:
                    self.logger.error(f"Ошибка в цикле: {str(e)}", exc_info=True)
                    time.sleep(60)
                    
        except KeyboardInterrupt:
            self.logger.info("\nОстановка бота...")
        finally:
            self._shutdown()

    def _shutdown(self):
        """Корректное завершение работы бота"""
        try:
            # Сохраняем последнее состояние
            self.update_bot_state()
            
            # Обновляем статус в файле состояния
            with open('bot_state.json', 'r+', encoding='utf-8') as f:
                state = json.load(f)
                state['status'] = 'stopped'
                state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.seek(0)
                json.dump(state, f, indent=4, ensure_ascii=False)
                f.truncate()
            
            self.logger.info("Бот корректно остановлен")
            
        except Exception as e:
            self.logger.error(f"Ошибк при остановке бота: {e}", exc_info=True)

    def _fetch_and_analyze_data(self):
        """Получение и анализ данных"""
        try:
            # Получаем данные
            if not self.fetch_data():
                self.logger.error("Не удалось получить данные с биржи")
                return False

            # Проверяем наличие данных
            if self.data is None or self.data.empty:
                self.logger.error("Данные не получены или пустые")
                return False

            # Создаем объект индикаторов и рассчитываем их
            self.indicators = TechnicalIndicators(self.data)
            self.data = self.indicators.calculate_all_indicators()

            # Генерируем сигнал
            signal = self.generate_signal()
            print(f"Сгенерирован сигнал: {signal}")

            # Если есть сигнал, рассчитываем метрики и добавляем в очереь
            if signal != 'HOLD':
                metrics = self.calculate_position_metrics(signal)
                self.signal_queue.put((signal, metrics))

            return True

        except Exception as e:
            self.logger.error(f"Ошибка при получении и анализе данных: {str(e)}", exc_info=True)
            return False

    def _process_signals_loop(self):
        """Бесконечный цикл обработки сигналов"""
        while True:
            try:
                signal, metrics = self.signal_queue.get()
                if signal and metrics:
                    # Выполняем торговую операцию в отдельном потоке
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        trade_future = executor.submit(self.execute_trade, signal, metrics)
                        if trade_future.result():
                            self.save_trade(signal, metrics)
                            self.print_analysis(signal, metrics)
            except Exception as e:
                self.logger.error(f"Ошибка обработки сигнала: {str(e)}", exc_info=True)
            finally:
                self.signal_queue.task_done()

    def _update_state_loop(self):
        """Бесконечный цикл обновления состояния"""
        while True:
            try:
                self.update_bot_state()
                time.sleep(30)  # Обновляем состояние каждые 30 секунд
            except Exception as e:
                self.logger.error(f"Ошибка обновления состояния: {str(e)}", exc_info=True)
                time.sleep(5)

    def run(self):
        """Запуск торгового бота"""
        try:
            print("\nЗапук торгового бота...")

            # олучение данных
            if not self.fetch_data():
                print("Не удалось полчить данные")
                return

            # Расчет иникаторов
            self.calculate_indicators()

            # Генерация сигнала
            signal = self.generate_signal()
            print(f"Сгенерирован сигнал: {signal}")

            # Рсчет метрик позиции
            metrics = self.calculate_position_metrics(signal) if signal != 'HOLD' else None

            # Сохранение сделки
            if metrics:
                self.save_trade(signal, metrics)

            # Вывд анализа
            self.print_analysis(signal, metrics)

        except Exception as e:
            print(f"Ошибка при работе бота: {e}")

    def _update_balance(self):
        """Обновление баланса"""
        try:
            if self.client:
                account = self.client.get_account()
                usdt_balance = float([asset for asset in account['balances']
                                   if asset['asset'] == 'USDT'][0]['free'])
                self.balance = usdt_balance
                self.logger.info(f"Баланс обновлен: {self.balance} USDT")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Ошибка обновления баланса: {str(e)}")
            return False

    def _save_error_state(self, error_message):
        """Сохранение состояния ошибки"""
        try:
            error_state = {
                'status': 'error',
                'error': str(error_message),
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'balance': self.balance,
                'current_price': float(
                    self.data['close'].iloc[-1]) if self.data is not None and not self.data.empty else None,
                'last_signal': self.last_signal,
                'technical_indicators': {
                    'rsi': None,
                    'sma20': None,
                    'sma50': None,
                    'bb_high': None,
                    'bb_low': None,
                    'bb_mid': None,
                    'price_change_24h': None,
                    'volume_24h': None,
                    'trend': None
                },
                'market_conditions': {
                    'volatility_level': None,
                    'volume_analysis': None,
                    'trend_reliability': None,
                    'market_phase': None
                }
            }

            with open('bot_state.json', 'w') as f:
                json.dump(error_state, f, indent=4)

            logging.error(f"Saved error state: {error_message}")

        except Exception as e:
            logging.critical(f"Failed to save error state: {e}")

    def _calculate_price_change_24h(self):
        """Расчет изменения цены за 24 часа"""
        try:
            if self.data is not None and len(self.data) >= 24:
                current_price = self.data['close'].iloc[-1]
                price_24h_ago = self.data['close'].iloc[-24]
                return ((current_price - price_24h_ago) / price_24h_ago * 100)
        except Exception as e:
            logging.error(f"Error calculating price change: {e}")
            return None

    def _calculate_support_level(self):
        """Расчет уровня поддержки"""
        try:
            if self.data is not None:
                # Используем минимумы последних 20 свечей
                return float(self.data['low'].tail(20).min())
            return None
        except Exception as e:
            logging.error(f"Error calculating support level: {e}")
            return None

    def _calculate_resistance_level(self):
        """Расчет уровня сопротивленя"""
        try:
            if self.data is not None:
                # Используем максимумы последних 20 свечей
                return float(self.data['high'].tail(20).max())
            return None
        except Exception as e:
            logging.error(f"Error calculating resistance level: {e}")
            return None

    def _analyze_volume(self):
        """Анализ объема торгов"""
        try:
            if self.data is not None:
                current_volume = self.data['volume'].iloc[-1]
                avg_volume = self.data['volume'].tail(24).mean()

                if current_volume > avg_volume * 1.5:
                    return 'high'
                elif current_volume < avg_volume * 0.5:
                    return 'low'
                else:
                    return 'normal'
            return None
        except Exception as e:
            logging.error(f"Error analyzing volume: {e}")
            return None

    def calculate_indicators(self, symbol):
        """Расчет всех технических индикаторов для символа"""
        try:
            if symbol not in self.market_data or not self.market_data[symbol]:
                raise ValueError(f"Нет данных для расчета индикаторов по {symbol}")
            
            market_data = self.market_data[symbol]
            
            # Преобразуем данные в DataFrame
            df = pd.DataFrame(market_data)
            
            # Создаем объект индикаторов
            indicators = TechnicalIndicators(df)
            
            # Рассчитываем индикаторы
            df_with_indicators = indicators.calculate_all_indicators()
            
            # Обновляем данные
            self.market_data[symbol] = df_with_indicators.to_dict('records')
            
            self.logger.info(f"Индикаторы успешно рассчитаны для {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при расчете индикаторов для {symbol}: {str(e)}", exc_info=True)
            return False

    def _format_trade_info(self, signal, metrics):
        """Форматирование информации о сделке"""
        try:
            # Расчет процентных значений
            stop_loss_percent = ((metrics['stop_loss'] - metrics['entry_price']) / metrics['entry_price'] * 100)
            take_profit_percent = ((metrics['take_profit'] - metrics['entry_price']) / metrics['entry_price'] * 100)
            risk_reward_ratio = abs(take_profit_percent / stop_loss_percent)

            trade_info = (
                f"\nТОРГОВЫЙ СИГНАЛ | {signal}\n"
                f"{'='*50}\n"
                f"\nПАРАМЕРЫ СДЕЛКИ:\n"
                f"- Тип сделки: {'ПОКУПКА' if signal == 'BUY' else 'ПРОДАЖА'}\n"
                f"- Цена входа: {metrics['entry_price']:.2f} USDT\n"
                f"- Размер позиции: {metrics['position_size']:.6f} BTC\n"
                f"- Объем в USDT: {metrics['position_size'] * metrics['entry_price']:.2f}\n"
                f"\nУПРАВЛЕНИЕ РИСКАМИ:\n"
                f"- Стоп-лосс: {metrics['stop_loss']:.2f} ({stop_loss_percent:+.2f}%)\n"
                f"- Тейк-профит: {metrics['take_profit']:.2f} ({take_profit_percent:+.2f}%)\n"
                f"- Риск/Прибыль: 1:{risk_reward_ratio:.2f}\n"
                f"- Риск в USDT: {metrics['risk_amount']:.2f}\n"
                f"- Потенциальная прибыль: {metrics['potential_profit']:.2f} USDT\n"
                f"\nБЛАНС СЧЕТА:\n"
                f"- Доступный баланс: {self.balance:.2f} USDT\n"
                f"- Использование баланса: {(metrics['position_size'] * metrics['entry_price'] / self.balance * 100):.2f}%\n"
                f"{'='*50}"
            )
            return trade_info
        except Exception as e:
            self.logger.error(f"Ошибка форматирования торговой информации: {str(e)}", exc_info=True)
            return None

    def _get_trading_statistics(self):
        """Получение статистики торговли"""
        try:
            total_trades = len(self.trades)
            if total_trades == 0:
                return None

            successful_trades = len([t for t in self.trades if t.get('profit', 0) > 0])
            total_profit = sum(t.get('profit', 0) for t in self.trades)
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Получам последние 5 сделок
            recent_trades = self.trades[-5:] if len(self.trades) >= 5 else self.trades
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'failed_trades': total_trades - successful_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'recent_trades': recent_trades
            }
        except Exception as e:
            self.logger.error(f"Ошибка получения статистики торговли: {str(e)}", exc_info=True)
            return None

    def _get_open_positions(self):
        """Получение открытых позиций"""
        try:
            positions = {}
            if not self.client:
                return positions
            
            for symbol in config.TRADING_PARAMS['symbols']:
                try:
                    # Получаем информацию о балансе
                    account = self.client.get_account()
                    base_asset = symbol[:-4]  # Убираем 'USDT' из конца
                    
                    asset_balance = next(
                        (asset for asset in account['balances'] if asset['asset'] == base_asset),
                        None
                    )
                    
                    if asset_balance:
                        free_amount = float(asset_balance['free'])
                        locked_amount = float(asset_balance['locked'])
                        total_amount = free_amount + locked_amount
                        
                        if total_amount > 0:
                            # Получаем текущую цену
                            ticker = self.client.get_symbol_ticker(symbol=symbol)
                            current_price = float(ticker['price'])
                            
                            # Рассчитываем стоимость позиции
                            value_usdt = total_amount * current_price
                            
                            positions[symbol] = {
                                'symbol': symbol,
                                'size': total_amount,
                                'value': value_usdt,
                                'entry_price': current_price,
                                'current_price': current_price,
                                'pnl': 0  # Можно добавить расчет PnL если нужно
                            }
                            
                except Exception as e:
                    self.logger.error(f"Ошибка получения позиции для {symbol}: {str(e)}")
                    continue
                
            return positions
            
        except Exception as e:
            self.logger.error(f"Ошибка получения открытых позиций: {str(e)}")
            return {}

    def initialize(self):
        """Инициализация бота"""
        self.logger.info("Инициализация бота...")
        
        # Инициализируем стратегию с передачей баланса
        self.strategy = GOATStrategy(
            risk_reward_ratio=2.0,
            max_loss=0.02,
            balance=self.balance
        )
        
        try:
            # Получаем список активов для торговли
            symbols = self.strategy.analyze_market(self.client)
            if not symbols:
                raise ValueError("Не удалось получить список активов для торговли")
            
            self.logger.info(f"Выбраны активы для торговли: {symbols}")
            
            # Обновляем конфигурацию
            config.TRADING_PARAMS['symbols'] = symbols
            
            # Инициализируем начальные данные для каждого символа
            for symbol in symbols:
                self.update_market_data(symbol)
                
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации: {str(e)}")
            raise

    def update_market_data(self, symbol):
        """Обновление рыночных данных для конкретного символа"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=config.TRADING_PARAMS['interval'],
                limit=config.TRADING_PARAMS['limit']
            )
            
            # Преобразование данных в нужный формат
            self.market_data[symbol] = [
                {
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                }
                for k in klines
            ]
        except Exception as e:
            self.logger.error(f"Ошибка при получении данных для {symbol}: {str(e)}")
            raise

    def check_and_execute_signals(self):
        """Проверка и исполнение торговых сигналов"""
        for symbol in config.TRADING_PARAMS['symbols']:
            try:
                if symbol not in self.market_data:
                    continue
                    
                # Сначала рассчитываем индикаторы
                if not self.calculate_indicators(symbol):
                    continue
                    
                signal = self.strategy.generate_signal(
                    self.market_data[symbol],
                    symbol
                )
                
                if signal['signal'] not in ['hold', 'HOLD']:
                    metrics = self.calculate_position_metrics(signal['signal'], symbol)
                    if metrics:
                        self.execute_trade(signal['signal'], metrics)
                
            except Exception as e:
                self.logger.error(f"Ошибка при обработке сигнала для {symbol}: {str(e)}")

    def process_signals(self, signals):
        """Обработка и вывод торговых сигналов"""
        self.logger.info("\nОтобранные активы для торговли:")
        self.logger.info("-" * 50)
        
        for signal in signals:
            self.logger.info(f"\nАктив: {signal['symbol']}")
            self.logger.info(f"Направление: {'LONG' if signal['direction'] == 'buy' else 'SHORT'}")
            self.logger.info(f"Точка входа: {signal['entry_price']:.8f}")
            self.logger.info(f"Стоп-лосс: {signal['stop_loss']:.8f} ({signal['stop_loss_percent']:.2f}%)")
            self.logger.info(f"Тейк-профит: {signal['take_profit']:.8f} ({signal['take_profit_percent']:.2f}%)")
            
            # Используем метод стратегии для расчета размера позиции
            position_size = self.strategy.calculate_position_size(
                signal['entry_price'], 
                signal['stop_loss'],
                self.balance
            )
            position_value = position_size * signal['entry_price']
            
            self.logger.info(f"Размер позиции: {position_size:.8f}")
            self.logger.info(f"Стоимость позиции: {position_value:.2f} USDT ({(position_value/self.balance*100):.2f}% от баланса)")
            
            potential_loss = abs(signal['entry_price'] - signal['stop_loss']) * position_size
            potential_profit = abs(signal['take_profit'] - signal['entry_price']) * position_size
            
            self.logger.info(f"Потенциальный убыток: {potential_loss:.2f} USDT")
            self.logger.info(f"Потенциальная прибыль: {potential_profit:.2f} USDT")
            self.logger.info(f"Соотношение риск/прибыль: 1:{potential_profit/potential_loss:.2f}")
            self.logger.info("-" * 30)

    def calculate_position_size(self, entry_price, stop_loss):
        """Расчет размера позиции"""
        try:
            if self.balance is None:
                self.logger.error("аланс не определен")
                return 0
            
            # Получаем риск на сделку из конфигурации (по умолчанию 1%)
            risk_per_trade = config.TRADING_PARAMS.get('risk_percentage', 1) / 100
            
            # Рассчитываем размер риска в USDT
            risk_amount = self.balance * risk_per_trade
            
            # Рассчитываем размер позиции исходя из риска
            position_size = risk_amount / abs(entry_price - stop_loss)
            
            # Проверяем минимальный размер позиции
            min_position_value = config.RISK_MANAGEMENT['position_sizing'].get('min_position_value', 50)
            position_value = position_size * entry_price
            
            if position_value < min_position_value:
                self.logger.warning(f"Размер позиции меньше минимального: {position_value:.2f} USDT (мин: {min_position_value} USDT)")
                return 0
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета размера позиции: {str(e)}")
            return 0


if __name__ == "__main__":
    bot = TradingBot()
    bot.run_auto_trading()  # Запускаем автоматическую торговлю вместо однократного анализа

