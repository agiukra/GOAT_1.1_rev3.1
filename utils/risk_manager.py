import logging
import config
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, exchange_handler):
        self.exchange = exchange_handler
        self.logger = logging.getLogger('TradingBot.RiskManager')
        self.daily_trades = {}
        self.open_positions = {}
        
    def reset_daily_stats(self):
        """Сброс дневной статистики"""
        self.daily_trades = {}
        
    def can_open_position(self, symbol, side, quantity, price):
        """Проверка возможности открытия позиции"""
        # Проверка максимального количества открытых позиций
        if len(self.open_positions) >= config.RISK_MANAGEMENT['risk_limits']['max_open_positions']:
            self.logger.warning("Достигнут лимит открытых позиций")
            return False
            
        # Проверка дневного лимита сделок
        today = datetime.now().date()
        if today not in self.daily_trades:
            self.daily_trades[today] = 0
        if self.daily_trades[today] >= config.RISK_MANAGEMENT['risk_limits']['max_daily_trades']:
            self.logger.warning("Достигнут дневной лимит сделок")
            return False
            
        # Проверка размера позиции
        position_value = quantity * price
        account_balance = self.exchange.get_balance('USDT')
        
        if position_value < config.RISK_MANAGEMENT['position_sizing']['min_position_value']:
            self.logger.warning(f"Размер позиции слишком мал: {position_value} USDT")
            return False
            
        position_size_ratio = position_value / account_balance
        if position_size_ratio > config.RISK_MANAGEMENT['position_sizing']['max_position_size']:
            self.logger.warning(f"Размер позиции превышает максимально допустимый: {position_size_ratio*100}%")
            return False
            
        return True
        
    def calculate_position_size(self, symbol, price):
        """Расчет размера позиции"""
        account_balance = self.exchange.get_balance('USDT')
        risk_per_trade = account_balance * config.RISK_MANAGEMENT['risk_limits']['max_trade_risk']
        
        # Расчет стоп-лосса
        stop_loss_percent = config.RISK_MANAGEMENT['stop_loss']['default_percentage'] / 100
        max_loss_amount = price * stop_loss_percent
        
        # Расчет количества с учетом риска
        quantity = risk_per_trade / max_loss_amount
        
        # Проверка на минимальный размер позиции
        min_position_size = config.RISK_MANAGEMENT['position_sizing']['min_position_size']
        if quantity * price < min_position_size:
            quantity = min_position_size / price
            
        return quantity
        
    def calculate_stop_loss(self, symbol, entry_price, side):
        """Расчет уровня стоп-лосса"""
        stop_percent = config.RISK_MANAGEMENT['stop_loss']['default_percentage'] / 100
        if side == 'BUY':
            return entry_price * (1 - stop_percent)
        else:
            return entry_price * (1 + stop_percent)
            
    def calculate_take_profit(self, symbol, entry_price, side):
        """Расчет уровней тейк-профита"""
        targets = []
        for i, target_percent in enumerate(config.RISK_MANAGEMENT['take_profit']['scaling_targets']):
            if side == 'BUY':
                price = entry_price * (1 + target_percent/100)
            else:
                price = entry_price * (1 - target_percent/100)
            targets.append({
                'price': price,
                'reduce_percent': config.RISK_MANAGEMENT['take_profit']['position_reduce'][i]
            })
        return targets
        
    def update_position(self, symbol, order):
        """Обновление информации о позиции"""
        if order['side'] == 'BUY':
            if symbol not in self.open_positions:
                self.open_positions[symbol] = {
                    'entry_price': float(order['price']),
                    'quantity': float(order['executedQty']),
                    'side': 'LONG'
                }
            else:
                # Усреднение позиции
                current_pos = self.open_positions[symbol]
                new_quantity = current_pos['quantity'] + float(order['executedQty'])
                new_entry = ((current_pos['entry_price'] * current_pos['quantity']) + 
                           (float(order['price']) * float(order['executedQty']))) / new_quantity
                self.open_positions[symbol]['entry_price'] = new_entry
                self.open_positions[symbol]['quantity'] = new_quantity
        else:
            if symbol in self.open_positions:
                current_pos = self.open_positions[symbol]
                new_quantity = current_pos['quantity'] - float(order['executedQty'])
                if new_quantity <= 0:
                    del self.open_positions[symbol]
                else:
                    self.open_positions[symbol]['quantity'] = new_quantity
                    
        # Обновление дневной статистики
        today = datetime.now().date()
        if today not in self.daily_trades:
            self.daily_trades[today] = 0
        self.daily_trades[today] += 1 