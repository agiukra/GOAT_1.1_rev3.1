from flask import Flask, render_template, url_for
from datetime import datetime
import json
import os

app = Flask(__name__, static_folder='static')

def load_bot_state():
    """Загрузка состояния бота из файла"""
    try:
        if os.path.exists('bot_state.json'):
            with open('bot_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)
                return state
        return None
    except Exception as e:
        print(f"Ошибка при загрузке состояния бота: {e}")
        return None

def format_time_ago(timestamp_str):
    """Форматирование времени"""
    try:
        if not timestamp_str:
            return "неизвестно"
        
        last_update = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        diff = now - last_update
        
        minutes = diff.total_seconds() / 60
        if minutes < 60:
            return f"{int(minutes)} минут назад"
        elif minutes < 1440:  # меньше 24 часов
            hours = int(minutes / 60)
            return f"{hours} часов назад"
        else:
            days = int(minutes / 1440)
            return f"{days} дней назад"
    except Exception as e:
        print(f"Ошибка при форматировании времени: {e}")
        return "неизвестно"

@app.route('/')
def index():
    """Главная страница"""
    try:
        state = load_bot_state()
        if not state:
            return render_template('error.html', error="Не удалось загрузить состояние бота")

        # Словарь для перевода названий индикаторов
        indicator_names = {
            'rsi': 'RSI',
            'sma20': 'SMA 20',
            'sma50': 'SMA 50',
            'bb_high': 'Верхняя BB',
            'bb_low': 'Нижняя BB',
            'bb_mid': 'Средняя BB',
            'price_change_24h': 'Изменение цены (24ч)',
            'volume_24h': 'Объем (24ч)',
            'volatility': 'Волатильность',
            'volatility_24h': 'Волатильность (24ч)',
            'highest_24h': 'Максимум (24ч)',
            'lowest_24h': 'Минимум (24ч)',
            'trend': 'Тренд',
            'trend_strength': 'Сила тренда',
            'market_momentum': 'Моментум рынка',
            'rsi_signal': 'Сигнал RSI',
            'bb_position': 'Позиция BB',
            'ma_cross': 'Пересечение MA',
            'momentum': 'Моментум'
        }

        # Словарь для перевода значений индикаторов
        value_translations = {
            'Medium': 'Средний',
            'Normal': 'Нормальный',
            'Uptrend': 'Восходящий',
            'Downtrend': 'Нисходящий',
            'Low': 'Низкий',
            'High': 'Высокий',
            'up': 'Восходящий',
            'down': 'Нисходящий',
            'overbought': 'Перекуплен',
            'oversold': 'Перепродан',
            'neutral': 'Нейтральный',
            '1': 'Да',
            '-1': 'Нет',
            '0': 'Нет сигнала'
        }

        # Словарь для перевода рыночных условий
        market_conditions_names = {
            'volatility_level': 'Уровень волатильности',
            'volume_analysis': 'Анализ объема',
            'market_phase': 'Фаза рынка',
            'trend_reliability': 'Надежность тренда'
        }

        # Словарь для перевода торговой статистики
        trading_stats_names = {
            'total_trades': 'Всего сделок',
            'successful_trades': 'Успешных сделок',
            'failed_trades': 'Неудачных сделок',
            'win_rate': 'Процент успешных',
            'total_profit': 'Общая прибыль'
        }

        current_time = datetime.now()
        last_update = state.get('last_update', '')
        time_ago = format_time_ago(last_update)

        # Форматируем технические индикаторы
        technical_indicators = {}
        for k, v in state.get('technical_indicators', {}).items():
            key = indicator_names.get(k, k)
            if isinstance(v, (int, float)):
                value = float(v)
            else:
                value = value_translations.get(str(v), v)
            technical_indicators[key] = value

        # Форматируем данные для отображения
        formatted_data = {
            'status': 'Активен' if state.get('status') == 'active' else 'Неактивен',
            'last_update': f"{last_update} ({time_ago})",
            'balance': float(state.get('balance', 0) or 0),
            'current_price': float(state.get('current_price', 0) or 0),
            'last_signal': {
                'BUY': 'ПОКУПКА',
                'SELL': 'ПРОДАЖА',
                'HOLD': 'ОЖИДАНИЕ'
            }.get(state.get('last_signal', 'HOLD'), 'ОЖИДАНИЕ'),
            'technical_indicators': technical_indicators,
            'market_conditions': {
                market_conditions_names.get(k, k): value_translations.get(str(v), v)
                for k, v in state.get('market_conditions', {}).items()
            },
            'risk_metrics': {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in state.get('risk_metrics', {}).items()
            },
            'open_positions': [
                {
                    'asset': pos.get('asset', ''),
                    'amount': float(pos.get('amount', 0) or 0),
                    'value_usdt': float(pos.get('value_usdt', 0) or 0)
                }
                for pos in state.get('open_positions', [])
            ],
            'trading_stats': {
                trading_stats_names.get(k, k): float(v) if isinstance(v, (int, float)) else v
                for k, v in state.get('trading_stats', {}).items()
            }
        }

        return render_template(
            'index.html',
            data=formatted_data,
            current_time=current_time.strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        print(f"Ошибка при форматировании данных: {e}")
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 