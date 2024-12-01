import logging
import time
from datetime import datetime
import config

class OrderExecutor:
    def __init__(self, exchange_handler, risk_manager):
        self.exchange = exchange_handler
        self.risk_manager = risk_manager
        self.logger = logging.getLogger('TradingBot.OrderExecutor')
        
    def execute_signal(self, symbol, side, signal_price):
        """Исполнение торгового сигнала"""
        try:
            # Расчет размера позиции
            quantity = self.risk_manager.calculate_position_size(symbol, signal_price)
            
            # Проверка возможности открытия позиции
            if not self.risk_manager.can_open_position(symbol, side, quantity, signal_price):
                self.logger.warning(f"Невозможно открыть позицию по {symbol}")
                return None
                
            # Размещение основного ордера
            order = self.exchange.place_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity
            )
            
            if not order:
                self.logger.error(f"Ошибка размещения ордера для {symbol}")
                return None
                
            # Обновление информации о позиции
            self.risk_manager.update_position(symbol, order)
            
            # Расчет и установка стоп-лосса
            stop_loss = self.risk_manager.calculate_stop_loss(
                symbol,
                float(order['price']),
                side
            )
            
            stop_order = self.exchange.place_order(
                symbol=symbol,
                side='SELL' if side == 'BUY' else 'BUY',
                order_type='STOP_LOSS_LIMIT',
                quantity=quantity,
                price=stop_loss,
                stopPrice=stop_loss
            )
            
            # Расчет и установка тейк-профитов
            take_profit_levels = self.risk_manager.calculate_take_profit(
                symbol,
                float(order['price']),
                side
            )
            
            tp_orders = []
            for tp in take_profit_levels:
                tp_quantity = quantity * tp['reduce_percent']
                tp_order = self.exchange.place_order(
                    symbol=symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    order_type='LIMIT',
                    quantity=tp_quantity,
                    price=tp['price']
                )
                if tp_order:
                    tp_orders.append(tp_order)
                    
            return {
                'main_order': order,
                'stop_loss': stop_order,
                'take_profits': tp_orders
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при исполнении сигнала: {str(e)}")
            return None
            
    def close_position(self, symbol):
        """Закрытие позиции"""
        try:
            if symbol not in self.risk_manager.open_positions:
                self.logger.warning(f"Нет открытой позиции для {symbol}")
                return None
                
            position = self.risk_manager.open_positions[symbol]
            
            # Отмена всех открытых ордеров
            open_orders = self.exchange.get_open_orders(symbol)
            for order in open_orders:
                self.exchange.cancel_order(symbol, order['orderId'])
                
            # Закрытие позиции по рынку
            close_order = self.exchange.place_order(
                symbol=symbol,
                side='SELL' if position['side'] == 'LONG' else 'BUY',
                order_type='MARKET',
                quantity=position['quantity']
            )
            
            if close_order:
                del self.risk_manager.open_positions[symbol]
                
            return close_order
            
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии позиции: {str(e)}")
            return None
            
    def modify_stop_loss(self, symbol, new_stop_price):
        """Модификация стоп-лосса"""
        try:
            # Отмена текущего стоп-лосса
            open_orders = self.exchange.get_open_orders(symbol)
            for order in open_orders:
                if order['type'] == 'STOP_LOSS_LIMIT':
                    self.exchange.cancel_order(symbol, order['orderId'])
                    
            # Установка нового стоп-лосса
            position = self.risk_manager.open_positions[symbol]
            new_stop_order = self.exchange.place_order(
                symbol=symbol,
                side='SELL' if position['side'] == 'LONG' else 'BUY',
                order_type='STOP_LOSS_LIMIT',
                quantity=position['quantity'],
                price=new_stop_price,
                stopPrice=new_stop_price
            )
            
            return new_stop_order
            
        except Exception as e:
            self.logger.error(f"Ошибка при модификации стоп-лосса: {str(e)}")
            return None 