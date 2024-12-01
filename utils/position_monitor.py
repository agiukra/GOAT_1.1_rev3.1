import logging
import time
from datetime import datetime
import threading
import config

class PositionMonitor:
    def __init__(self, exchange_handler, risk_manager, order_executor):
        self.exchange = exchange_handler
        self.risk_manager = risk_manager
        self.order_executor = order_executor
        self.logger = logging.getLogger('TradingBot.PositionMonitor')
        self.is_running = False
        self.monitor_thread = None
        
    def start(self):
        """Запуск мониторинга позиций"""
        if self.is_running:
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Мониторинг позиций запущен")
        
    def stop(self):
        """Остановка мониторинга позиций"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Мониторинг позиций остановлен")
        
    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        while self.is_running:
            try:
                self._check_positions()
                time.sleep(1)  # Проверка каждую секунду
            except Exception as e:
                self.logger.error(f"Ошибка в цикле мониторинга: {str(e)}")
                
    def _check_positions(self):
        """Проверка открытых позиций"""
        for symbol in list(self.risk_manager.open_positions.keys()):
            position = self.risk_manager.open_positions[symbol]
            
            # Получение текущей цены
            ticker = self.exchange.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # Проверка трейлинг-стопа
            if config.RISK_MANAGEMENT['stop_loss']['trailing_stop']:
                self._update_trailing_stop(symbol, position, current_price)
                
            # Проверка условий закрытия позиции
            if self._check_close_conditions(symbol, position, current_price):
                self.order_executor.close_position(symbol)
                
    def _update_trailing_stop(self, symbol, position, current_price):
        """Обновление трейлинг-стопа"""
        try:
            trailing_distance = config.RISK_MANAGEMENT['stop_loss']['trailing_distance'] / 100
            
            if position['side'] == 'LONG':
                potential_profit = (current_price - position['entry_price']) / position['entry_price']
                if potential_profit > trailing_distance:
                    new_stop = current_price * (1 - trailing_distance)
                    self.order_executor.modify_stop_loss(symbol, new_stop)
                    
            else:  # SHORT
                potential_profit = (position['entry_price'] - current_price) / position['entry_price']
                if potential_profit > trailing_distance:
                    new_stop = current_price * (1 + trailing_distance)
                    self.order_executor.modify_stop_loss(symbol, new_stop)
                    
        except Exception as e:
            self.logger.error(f"Ошибка обновления трейлинг-стопа: {str(e)}")
            
    def _check_close_conditions(self, symbol, position, current_price):
        """Проверка условий для закрытия позиции"""
        try:
            # Проверка достижения стоп-лосса
            stop_loss = self.risk_manager.calculate_stop_loss(
                symbol,
                position['entry_price'],
                'BUY' if position['side'] == 'LONG' else 'SELL'
            )
            
            if position['side'] == 'LONG':
                if current_price <= stop_loss:
                    self.logger.info(f"Достигнут стоп-лосс для {symbol}")
                    return True
            else:
                if current_price >= stop_loss:
                    self.logger.info(f"Достигнут стоп-лосс для {symbol}")
                    return True
                    
            # Дополнительные условия закрытия можно добавить здесь
            
            return False
            
        except Exception as e:
            self.logger.error(f"Ошибка проверки условий закрытия: {str(e)}")
            return False
            
    def get_position_status(self, symbol):
        """Получение статуса позиции"""
        try:
            if symbol not in self.risk_manager.open_positions:
                return None
                
            position = self.risk_manager.open_positions[symbol]
            ticker = self.exchange.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            entry_price = position['entry_price']
            quantity = position['quantity']
            side = position['side']
            
            if side == 'LONG':
                pnl = (current_price - entry_price) * quantity
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl = (entry_price - current_price) * quantity
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
                
            return {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'current_price': current_price,
                'quantity': quantity,
                'pnl': pnl,
                'pnl_percent': pnl_percent
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения статуса позиции: {str(e)}")
            return None 