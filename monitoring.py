from flask import Flask, render_template, jsonify, request, Response
from datetime import datetime
import logging
import traceback
from functools import wraps
from config import MONITOR_AUTH_USERNAME, MONITOR_AUTH_PASSWORD

# Настройка логирования в файл
logging.basicConfig(
    filename='monitor.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Инициализация Flask
app = Flask(__name__, template_folder='templates')
trading_bot = None
strategy = None

# Настройки аутентификации
AUTH_USERNAME = 'admin'
AUTH_PASSWORD = 'admin'

def check_auth(username, password):
    """Проверка авторизации"""
    return username == MONITOR_AUTH_USERNAME and password == MONITOR_AUTH_PASSWORD

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response('Необходима авторизация', 401,
                {'WWW-Authenticate': 'Basic realm="Вход в систему"'})
        return f(*args, **kwargs)
    return decorated

@app.route('/')
@requires_auth
def index():
    """Главная страница мониторинга"""
    global trading_bot
    
    try:
        monitor_data = {
            'status': 'Неактивен',
            'last_update': None,
            'balance': 0.00,
            'current_price': 0.00,
            'last_signal': 'ОЖИДАНИЕ',
            'trades_today': 0,
            'total_trades': 0,
            'open_positions': 0,
            'recent_trades': [],
            'error_message': None
        }
        
        if trading_bot and trading_bot.client:
            try:
                ticker = trading_bot.client.get_symbol_ticker(symbol='BTCUSDT')
                monitor_data.update({
                    'status': 'Активен',
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'balance': round(trading_bot.balance, 2),
                    'current_price': float(ticker['price']),
                    'last_signal': trading_bot.last_signal,
                    'trades_today': len([t for t in trading_bot.trades if 
                        datetime.fromisoformat(t['timestamp']).date() == datetime.now().date()]),
                    'total_trades': len(trading_bot.trades),
                    'open_positions': len(trading_bot._monitor_positions()),
                    'recent_trades': trading_bot.trades[-10:] if trading_bot.trades else []
                })
            except Exception as e:
                monitor_data['error_message'] = str(e)
                logging.error(f"Ошибка при получении данных: {str(e)}")
        
        return render_template('monitor.html', monitor=monitor_data)
    except Exception as e:
        logging.error(f"Критическая ошибка в index(): {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/api/status')
@requires_auth
def get_status():
    """API endpoint для получения статуса"""
    try:
        if not trading_bot or not trading_bot.client:
            raise ValueError("Торговый бот не инициализирован")
            
        # Получаем текущую цену BTC
        ticker = trading_bot.client.get_symbol_ticker(symbol='BTCUSDT')
        current_price = float(ticker['price'])
        
        # Получаем статус бота
        status = 'Активен' if trading_bot.client and trading_bot.balance > 0 else 'Неактивен'
        
        return jsonify({
            'status': status,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'balance': round(trading_bot.balance, 2),
            'current_price': current_price,
            'last_signal': trading_bot.last_signal,
            'today_trades': len([t for t in trading_bot.trades if 
                datetime.fromisoformat(t['timestamp']).date() == datetime.now().date()]),
            'total_trades': len(trading_bot.trades),
            'open_positions': len(trading_bot._monitor_positions())
        })
    except Exception as e:
        logging.error(f"Ошибка в get_status: {str(e)}")
        return jsonify({
            'status': 'Ошибка',
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/api/trades')
@requires_auth
def get_trades():
    """API endpoint для получения сделок"""
    global trading_bot
    if trading_bot is None:
        return jsonify([])
        
    try:
        # Возвращаем последние 10 сделок
        trades = trading_bot.trades[-10:]
        return jsonify(trades)
    except Exception as e:
        logging.error(f"Ошибка получения сделок: {str(e)}")
        return jsonify([])

@app.route('/conditions')
@requires_auth
def trading_conditions():
    """API endpoint для получения текущих торговых условий"""
    if strategy is None:
        return jsonify({'error': 'Стратегия не инициализирована'})
        
    try:
        conditions = strategy.get_current_conditions()
        return render_template('conditions.html', conditions=conditions)
    except Exception as e:
        logging.error(f"Ошибка получения торговых условий: {str(e)}")
        return jsonify({'error': str(e)})

def start_monitoring(bot, trading_strategy):
    """
    Запуск веб-монитора
    
    Args:
        bot: Торговый бот
        trading_strategy: Торговая стратегия
    """
    global trading_bot, strategy
    trading_bot = bot
    strategy = trading_strategy
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=False)
    except Exception as e:
        logging.error(f"Ошибка запуска монитора: {str(e)}")

if __name__ == '__main__':
    # Создаем тестовый бот для демонстрации
    class TestBot:
        def __init__(self):
            self.client = None
            self.balance = 0.0
            self.trades = []
            self.last_signal = 'HOLD'
            
        def _monitor_positions(self):
            return []
            
    class TestStrategy:
        def get_current_conditions(self):
            return {
                'rsi_value': 50,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_short': 100,
                'ema_medium': 95,
                'ema_long': 90,
                'current_trend': 'SIDEWAYS',
                'volume_surge': False,
                'volatility': 1.5
            }
    
    test_bot = TestBot()
    test_strategy = TestStrategy()
    
    # Запускаем веб-сервер с тестовыми данными
    start_monitoring(test_bot, test_strategy) 