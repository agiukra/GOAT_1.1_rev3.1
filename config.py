# Binance API credentials
API_KEY = 'xIWmr6Kgk3CYclvd4F8QSvoERyKsYt7YKVAXAq33RF5DIyYTBQ7ueMjto47bfF7m'
API_SECRET = 'OcybRBPZ89EA1dx8ZrLs8hGgfJ5axeBg0tNT4fwQb71IhKv1C8dJo8YtldIgSetC'

# Основные торговые параметры
TRADING_PARAMS = {
    'symbols': ['BTCUSDT', 'ETHUSDT'],  # Базовые пары
    'interval': '1m',                    # Таймфрейм
    'limit': 100,                        # Количество свечей для анализа
    'cycle_interval': 60,                # Интервал между циклами в секундах
    'interval_seconds': 300,
    'risk_percentage': 1,
    'account_balance': None,
    'allocation': {},  # Будет заполняться динамически
    'dynamic_allocation': True,  # Включаем динамическое распределение
    'min_order_value': 10,  # Минимальная стоимость ордера в USDT
    'max_order_value': 1000  # Максимальная стоимость ордера в USDT
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
        'rsi_lower': 25,      # Нижняя граница RSI для покупки
        'rsi_upper': 75,      # Верхняя граница RSI для покупки
        'volume_factor': 0.8,  # Минимальный множитель объема
        'trend_strength': 0.01 # Минимальная сила тренда
    },
    'sell': {
        'rsi_lower': 25,
        'rsi_upper': 75,
        'volume_factor': 0.8,
        'trend_strength': 0.01
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
    'level': 'DEBUG',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'trading_bot.log',
    'max_file_size': 1024 * 1024,
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

# Добавить важные параметры безопасности
SAFETY_SETTINGS = {
    'max_daily_trades': 5,
    'max_open_positions': 2,
    'max_daily_drawdown': 0.05,  # 5% максимальная просадка в день
    'emergency_stop_loss': 0.15,  # 15% аварийный стоп
}

# Параметры для реальной торговли
PRODUCTION_SETTINGS = {
    'paper_trading': False,
    'use_real_money': True,
    'enable_emergency_stop': True,
    'enable_notifications': True,
}

# Добавляем специфичные параметры для каждой пары
SYMBOL_SPECIFIC_PARAMS = {
    'BTCUSDT': {
        'min_order_size': 0.001,
        'price_precision': 2,
        'quantity_precision': 5
    },
    'ETHUSDT': {
        'min_order_size': 0.01,
        'price_precision': 2,
        'quantity_precision': 4
    },
    'BNBUSDT': {
        'min_order_size': 0.1,
        'price_precision': 2,
        'quantity_precision': 3
    }
}

# Параметры отбора активов
ASSET_SELECTION = {
    'max_assets': 100,  # Сколько топовых активов анализировать
    'min_volume_24h': 1000000,  # Минимальный 24ч объем в USDT
    'min_market_cap': 10000000,  # Минимальная капитализация
    'min_volatility': 0.02,  # Минимальная волатильность (2%)
    'max_volatility': 0.15,  # Максимальная волатильность (15%)
    'min_price_usd': 0.1,   # Минимальная цена в USD
    'max_price_usd': 500,   # Максимальная цена в USD
    'blacklist': [
        'USDT', 'BUSD', 'USDC', 'DAI', 'PAX', 'TUSD', 'USDN',
        'USDP', 'USDD', 'FDUSDT', 'FDUSD'  # Стейблкоины
    ],
    'selection_criteria': {
        'volatility_weight': 0.3,
        'volume_weight': 0.3,
        'trend_weight': 0.4,
        'max_selected_assets': 5
    },
    'update_interval': 24,
    'min_price': 0.1,
    'max_spread': 0.02,
    'quote_currency': 'USDT',
    'min_market_rank': 1,    # Минимальный ранг по капитализации
    'max_market_rank': 300   # Максимальный ранг по капитализации
}

# Добавьте эти параметры в конфигурацию
STOP_LOSS_PERCENT = 2.0  # Стоп-лосс 2%
TAKE_PROFIT_PERCENT = 4.0  # Тейк-профит 4%

# Добавить настройки RSI в конфигурацию
RSI_SETTINGS = {
    'period': 14,
    'oversold': 30,
    'overbought': 75  # Увеличить с 70 до 75
}

# Добавляем параметры размера позиции
POSITION_SIZING = {
    'risk_per_trade': 0.01,        # 1% риска на сделку
    'max_position_percent': 0.02,   # Максимум 2% от баланса
    'min_position_size': 10,       # Минимум 10 USDT
    'max_position_size': None      # Будет рассчитано как % от баланса
}