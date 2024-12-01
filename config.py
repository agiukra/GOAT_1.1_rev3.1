# Binance API credentials
API_KEY = 'xIWmr6Kgk3CYclvd4F8QSvoERyKsYt7YKVAXAq33RF5DIyYTBQ7ueMjto47bfF7m'
API_SECRET = 'OcybRBPZ89EA1dx8ZrLs8hGgfJ5axeBg0tNT4fwQb71IhKv1C8dJo8YtldIgSetC'

# Основные торговые параметры
TRADING_PARAMS = {
    'symbol': 'BTCUSDT',
    'interval': '1m',  # Используем минутный таймфрейм для HFT
    'limit': 100,
    'interval_seconds': 60,  # Проверка каждую минуту
    'risk_percentage': 1,
    'account_balance': None,  # Будет получено автоматически
    'max_pairs': 1,  # Торгуем только одной парой
    'min_daily_volume': 1000000,  # Минимальный дневной объем в USDT
    'min_volatility': 0.5,  # Минимальная волатильность в %
    'max_volatility': 5.0,  # Максимальная волатильность в %
}

# Параметры индикаторов
INDICATOR_PARAMS = {
    'rsi': {
        'period': 2,
        'overbought': 70,
        'oversold': 30
    },
    'bollinger_bands': {
        'period': 20,
        'std_dev': 2
    },
    'moving_averages': {
        'short_period': 2,
        'long_period': 4
    }
}

# Условия для торговых сигналов
SIGNAL_CONDITIONS = {
    'buy': {
        'rsi_threshold': 30,
        'bb_position': -1,
        'trend_confirmation': True,
        'volume_threshold': 1.5,
        'min_price_change': 0.1,
        'momentum_threshold': 0.2
    },
    'sell': {
        'rsi_threshold': 70,
        'bb_position': 1,
        'trend_confirmation': True,
        'volume_threshold': 1.5,
        'min_price_change': 0.1,
        'momentum_threshold': -0.2
    }
}

# Управление рисками
RISK_MANAGEMENT = {
    'position_sizing': {
        'max_position_size': 0.1,  # 10% от баланса
        'min_position_size': 0.01,  # 1% от баланса
        'min_position_value': 50,  # Минимальная стоимость позиции в USDT
        'dust_threshold': 10,      # Порог для определения "пыли" в USDT
        'position_scaling': False  # Отключаем масштабирование для HFT
    },
    'stop_loss': {
        'default_percentage': 0.5,  # 0.5% от цены входа
        'trailing_stop': True,
        'trailing_distance': 0.2,  # 0.2% от текущей цены
        'atr_multiplier': 1.0      # Минимальный множитель для динамического стоп-лосса
    },
    'take_profit': {
        'default_percentage': 1.0,  # 1% от цены входа
        'scaling_targets': [0.5, 0.8, 1.0],  # Близкие цели по прибыли
        'position_reduce': [0.4, 0.3, 0.3]   # Процент закрытия на каждой цели
    },
    'risk_limits': {
        'max_daily_risk': 0.05,    # 5% от баланса
        'max_trade_risk': 0.01,    # 1% от баланса на сделку
        'max_open_positions': 1,   # Только одна открытая позиция
        'max_daily_trades': 100,   # Максимум 100 сделок в день
        'min_balance': 100         # Минимальный баланс для торговли
    }
}

# Параметры анализа рынка
MARKET_ANALYSIS = {
    'volatility': {
        'high_threshold': 2.0,
        'low_threshold': 0.2,
        'period': 24
    },
    'trend': {
        'strength_threshold': 0.01,
        'min_confirmation_periods': 2,
        'momentum_period': 3
    },
    'volume': {
        'high_threshold': 1.5,
        'low_threshold': 0.5,
        'average_period': 24
    },
    'support_resistance': {
        'lookback_periods': 20,
        'min_touches': 2,
        'price_threshold': 0.01
    }
}

# Настройки логирования
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'trading_bot.log',
    'max_file_size': 1024 * 1024,  # 1MB
    'backup_count': 5
}

# Настройки уведомлений
NOTIFICATIONS = {
    'enabled': True,
    'telegram': {
        'enabled': False,
        'bot_token': 'your_telegram_bot_token',
        'chat_id': 'your_chat_id'
    },
    'email': {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': 'your_email@gmail.com',
        'sender_password': 'your_app_password'
    },
    'events': {
        'trade_executed': True,
        'signal_generated': True,
        'error_occurred': True,
        'daily_summary': True
    }
}

# Настройки бэктестинга
BACKTEST = {
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'initial_balance': 10000,
    'commission': 0.001,
    'slippage': 0.0005
}

# Добавьте или обновите следующие параметры
TRADING_STRATEGY = {
    'name': 'GOAT',
    'timeframe': '1m',
    'rsi_period': 2,
    'ema_periods': [2, 3, 4],
    'parameters': {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'ema_fast': 2,
        'ema_medium': 3,
        'ema_slow': 4
    }
}

# Убедитесь, что остальные параметры конфигурации присутствуют
TRADE_AMOUNT = 100  # сумма для торговли в USDT
MAX_POSITIONS = 1   # максимальное количество одновременных позиций