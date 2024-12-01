import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import yfinance as yf
from strategies.goat_strategy import GOATStrategy

def test_strategy():
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
    strategy = GOATStrategy(config)
    
    # Рассчитываем индикаторы
    indicators = strategy.calculate_indicators(data)
    
    # Генерируем сигналы
    signals = []
    for i in range(len(data)):
        if i < 50:  # Пропускаем первые дни для накопления данных индикаторов
            continue
        
        current_data = data.iloc[:i+1]
        current_indicators = {
            'rsi': indicators['rsi'].iloc[:i+1],
            'ema_short': indicators['ema_short'].iloc[:i+1],
            'ema_medium': indicators['ema_medium'].iloc[:i+1],
            'ema_long': indicators['ema_long'].iloc[:i+1],
            'atr': indicators['atr'].iloc[:i+1],
            'volume_ma': indicators['volume_ma'].iloc[:i+1]
        }
        
        signal = strategy.generate_signal(current_data, current_indicators)
        signals.append(signal)
        
        if signal != 'HOLD':
            print(f"Date: {data.index[i]}, Signal: {signal}, Price: {data['close'][i]:.2f}")
            if signal == 'BUY':
                stop_loss = strategy.calculate_stop_loss(current_data, current_indicators, 'BUY')
                take_profit = strategy.calculate_take_profit(current_data, current_indicators, 'BUY')
                position_size = strategy.calculate_position_size(current_data, current_indicators)
                print(f"Position Size: {position_size:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
            print("---")

    # Выводим статистику сигналов
    total_signals = len([s for s in signals if s != 'HOLD'])
    buy_signals = len([s for s in signals if s == 'BUY'])
    sell_signals = len([s for s in signals if s == 'SELL'])
    
    print("\nСтатистика сигналов:")
    print(f"Всего сигналов: {total_signals}")
    print(f"Сигналов на покупку: {buy_signals}")
    print(f"Сигналов на продажу: {sell_signals}")

if __name__ == '__main__':
    test_strategy() 