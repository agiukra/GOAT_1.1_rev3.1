import logging
from datetime import datetime
import numpy as np

class SafetyManager:
    """Менеджер безопасности торговли"""
    
    def __init__(self, config):
        self.config = config
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.initial_balance = config.get('initial_balance', 0)
        self.max_position_size = config.get('max_position_size', 0.4)
        self.max_daily_trades = config.get('max_daily_trades', 10)
        
        self.daily_trades = 0
        self.last_trade_reset = datetime.now().date()
        self.metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'current_balance': 0,
            'open_positions': {},
            'last_error': None,
            'drawdown_history': []
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        """Настройка системы логирования"""
        logging.basicConfig(
            filename='trading_bot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def pre_trade_checks(self, exchange, balance, position_size):
        """Проверки перед сделкой"""
        try:
            # Сброс дневного счетчика сделок
            current_date = datetime.now().date()
            if current_date > self.last_trade_reset:
                self.daily_trades = 0
                self.last_trade_reset = current_date
            
            # Проверка количества дневных сделок
            if self.daily_trades >= self.max_daily_trades:
                logging.warning("Превышен лимит дневных сделок")
                return False
                
            # Проверка размера позиции
            if position_size > balance * self.max_position_size:
                logging.warning(f"Слишком большой размер позиции: {position_size}")
                return False
                
            # Проверка просадки
            current_drawdown = self.calculate_drawdown(balance)
            if current_drawdown > self.max_drawdown:
                logging.warning(f"Превышена максимальная просадка: {current_drawdown:.2%}")
                return False
                
            # Проверка связи с биржей
            if not exchange.ping():
                logging.error("Нет связи с биржей")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Ошибка в pre_trade_checks: {str(e)}")
            return False
            
    def calculate_drawdown(self, current_balance):
        """Расчет текущей просадки"""
        if self.initial_balance == 0:
            return 0
        drawdown = (self.initial_balance - current_balance) / self.initial_balance
        self.metrics['drawdown_history'].append(drawdown)
        return drawdown
        
    def update_metrics(self, trade_result):
        """Обновление метрик торговли"""
        self.metrics['total_trades'] += 1
        if trade_result['success']:
            self.metrics['successful_trades'] += 1
        else:
            self.metrics['failed_trades'] += 1
            self.metrics['last_error'] = trade_result.get('error')
            
        self.daily_trades += 1
        
    def emergency_stop(self, reason):
        """Аварийная остановка торговли"""
        logging.critical(f"Аварийная остановка торговли: {reason}")
        # Здесь код для остановки торговли и уведомления
        return {'success': False, 'error': reason} 