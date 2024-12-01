# Binance API credentials
API_KEY = 'xIWmr6Kgk3CYclvd4F8QSvoERyKsYt7YKVAXAq33RF5DIyYTBQ7ueMjto47bfF7m'
API_SECRET = 'OcybRBPZ89EA1dx8ZrLs8hGgfJ5axeBg0tNT4fwQb71IhKv1C8dJo8YtldIgSetC'

# Основные торговые параметры
TRADING_PARAMS = {
    'symbol': 'BTCUSDT',
    'interval': '1h',
    'limit': 100,
    'interval_seconds': 300,
    'risk_percentage': 1,
    'account_balance': None
}

# Параметры индикаторов
INDICATOR_PARAMS = {
    'rsi': {
        'period': 14,
        'overbought': 70,
        'oversold': 30
    },
    'bollinger_bands': {
        'period': 20,
        'std_dev': 2
    },
    'moving_averages': {
        'short_period': 20,
        'long_period': 50
    }
}

# Условия для торговых сигналов
SIGNAL_CONDITIONS = {
    'buy': {
        'rsi_threshold': 30,
        'bb_position': -1,  # Цена ниже нижней полосы
        'trend_confirmation': True,  # Требуется подтверждение тренда
        'volume_threshold': 1.5,  # Объем выше среднего в 1.5 раза
        'min_price_change': 0.5,  # Минимальное изменение цены в %
        'momentum_threshold': 0.2
    },
    'sell': {
        'rsi_threshold': 70,
        'bb_position': 1,  # Цена выше верхней полосы
        'trend_confirmation': True,
        'volume_threshold': 1.5,
        'min_price_change': 0.5,
        'momentum_threshold': -0.2
    }
}

# Управление рисками
RISK_MANAGEMENT = {
    'position_sizing': {
        'max_position_size': 0.1,  # 10% от баланса
        'min_position_size': 0.01,  # 1% от баланса
        'min_position_value': 50,  # Минимальная стоимость позиции в USDT для отображения
        'dust_threshold': 10,      # Порог для определения "пыли" в USDT
        'position_scaling': True   # Разрешить масштабирование позиций
    },
    'stop_loss': {
        'default_percentage': 1.0,  # 1% от цены входа
        'trailing_stop': True,
        'trailing_distance': 0.5,  # 0.5% от текущей цены
        'atr_multiplier': 2.0      # Для динамического стоп-лосса
    },
    'take_profit': {
        'default_percentage': 3.0,  # 3% от цены входа
        'scaling_targets': [1.5, 2.0, 3.0],  # Множественные цели по прибыли
        'position_reduce': [0.3, 0.3, 0.4]   # Процент закрытия на каждой цели
    },
    'risk_limits': {
        'max_daily_risk': 0.02,    # 2% от баланса
        'max_trade_risk': 0.01,    # 1% от баланса на сделку
        'max_open_positions': 3,
        'max_daily_trades': 10,
        'min_balance': 100         # Минимальный баланс для торговли
    }
}

# Параметры анализа рынка
MARKET_ANALYSIS = {
    'volatility': {
        'high_threshold': 2.0,
        'low_threshold': 0.5,
        'period': 24
    },
    'trend': {
        'strength_threshold': 0.02,
        'min_confirmation_periods': 3,
        'momentum_period': 12
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