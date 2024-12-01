from flask import Flask, render_template, jsonify
import json
from datetime import datetime
import time
import threading

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Включаем поддержку Unicode в JSON
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Автоматическая перезагрузка шаблонов

def load_bot_state():
    """Загрузка состояния бота из файла"""
    try:
        # Добавляем кэширование состояния
        if hasattr(load_bot_state, 'cache'):
            cache = load_bot_state.cache
            if time.time() - cache['timestamp'] < 1:  # Кэш действителен 1 секунду
                return cache['state']
        
        with open('bot_state.json', 'r', encoding='utf-8') as f:
            state = json.load(f)
            current_time = datetime.now()
            last_update = datetime.strptime(state['last_update'], "%Y-%m-%d %H:%M:%S")
            time_diff = (current_time - last_update).total_seconds()
            
            # Форматируем время последнего обновления
            if time_diff < 60:
                time_str = f"{int(time_diff)} секунд назад"
            elif time_diff < 3600:
                minutes = int(time_diff // 60)
                time_str = f"{minutes} минут назад"
            else:
                hours = int(time_diff // 3600)
                time_str = f"{hours} часов назад"
            
            print(f"Состояние бота загружено: {current_time}")
            print(f"Последнее обновление: {state['last_update']} ({time_str})")
            
            # Определяем статус обновления
            if time_diff > 300:  # Более 5 минут
                state['update_status'] = 'warning'
                print(f"ВНИМАНИЕ: Данные устарели (более 5 минут)")
            elif time_diff > 600:  # Более 10 минут
                state['update_status'] = 'error'
                print(f"ОШИБКА: Данные сильно устарели (более 10 минут)")
            else:
                state['update_status'] = 'ok'
            
            state['time_since_update'] = time_diff
            state['time_since_update_str'] = time_str
            
            # Кэшируем состояние
            load_bot_state.cache = {
                'state': state,
                'timestamp': time.time()
            }
            
            return state
    except Exception as e:
        print(f"Ошибка чтения состояния: {e}")
        return None

@app.route('/')
def index():
    """Отображение главной страницы мониторинга"""
    state = load_bot_state()
    if not state:
        return render_template(
            'error.html',
            error="Бот не запущен или файл состояния недоступен"
        )

    try:
        # Проверяем актуальность данных
        current_time = datetime.now()
        last_update = datetime.strptime(state['last_update'], "%Y-%m-%d %H:%M:%S")
        time_diff = (current_time - last_update).total_seconds()
        
        # Добавляем информацию о времени обновления
        state['time_since_update'] = time_diff
        state['time_since_update_str'] = f"{int(time_diff)} секунд назад"
        
        # Форматируем данные для отображения
        market_data = {
            'current_price': state.get('current_price', 0),
            'price_change_24h': state.get('technical_indicators', {}).get('price_change_24h', 0),
            'volume_24h': state.get('technical_indicators', {}).get('volume_24h', 0),
            'trend': state.get('technical_indicators', {}).get('trend', 'unknown'),
            'volatility': state.get('technical_indicators', {}).get('volatility', 0),
            'rsi': state.get('technical_indicators', {}).get('rsi', 0),
            'sma20': state.get('technical_indicators', {}).get('sma20', 0),
            'sma50': state.get('technical_indicators', {}).get('sma50', 0),
            'bb_position': state.get('technical_indicators', {}).get('bb_position', 0),
            'momentum': state.get('technical_indicators', {}).get('market_momentum', 0)
        }

        # Получаем информацию о рыночных условиях
        market_conditions = state.get('market_conditions', {
            'volatility_level': 'unknown',
            'volume_analysis': 'unknown',
            'market_phase': 'unknown'
        })
        
        # Форматируем время последнего обновления
        last_update = datetime.strptime(state['last_update'], "%Y-%m-%d %H:%M:%S")
        time_since_update = (datetime.now() - last_update).total_seconds()
        
        return render_template(
            'index.html',
            state=state,
            market_data=market_data,
            market_conditions=market_conditions,
            last_update=last_update,
            time_since_update=time_diff
        )
    except Exception as e:
        print(f"Ошибка при форматировании данных: {e}")
        return render_template(
            'error.html',
            error=f"Ошибка обработки данных: {str(e)}"
        )

@app.route('/status')
def status():
    """API endpoint для получения статуса бота"""
    state = load_bot_state()
    if state:
        return jsonify(state)
    return jsonify({'error': 'Не удалось загрузить состояние бота'}), 500

@app.route('/api/market_data')
def market_data():
    """API endpoint для получения рыночных данных"""
    state = load_bot_state()
    if not state:
        return jsonify({'error': 'Не удалось загрузить состояние бота'}), 500
    
    return jsonify({
        'current_price': state.get('current_price'),
        'technical_indicators': state.get('technical_indicators'),
        'market_conditions': state.get('market_conditions')
    })

@app.route('/api/trading_stats')
def trading_stats():
    """API endpoint для получения торговой статистики"""
    state = load_bot_state()
    if not state:
        return jsonify({'error': 'Не удалось загрузить состояние бота'}), 500
    
    return jsonify({
        'balance': state.get('balance'),
        'last_signal': state.get('last_signal'),
        'risk_metrics': state.get('risk_metrics')
    })

@app.route('/switch_strategy/<strategy_name>')
def switch_strategy(strategy_name):
    """API endpoint для переключения стратегии"""
    try:
        state = load_bot_state()
        if not state:
            return jsonify({'error': 'Не удалось загрузить состояние бота'}), 500
        
        # Проверяем существование стратегии
        available_strategies = ['MACD', 'RSI_BB']
        if strategy_name not in available_strategies:
            return jsonify({
                'error': f'Неизвестная стратегия. Доступные стратегии: {", ".join(available_strategies)}'
            }), 400
        
        # Обновляем стратегию в состоянии
        state['current_strategy'] = strategy_name
        
        # Сохраняем обновленное состоя��ие
        with open('bot_state.json', 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
        
        return jsonify({
            'success': True, 
            'message': f'Стратегия изменена на {strategy_name}',
            'strategy': strategy_name
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Ошибка при переключении стратегии: {str(e)}'
        }), 500

def create_app():
    return app

def save_bot_state(state):
    """Сохранение состояния бота в файл"""
    try:
        with open('bot_state.json', 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Ошибка сохранения состояния: {e}")
        return False

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 