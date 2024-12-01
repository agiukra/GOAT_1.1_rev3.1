from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
import config
import time
from decimal import Decimal, ROUND_DOWN

class ExchangeHandler:
    def __init__(self):
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.logger = logging.getLogger('TradingBot.ExchangeHandler')
        
    def get_symbol_info(self, symbol):
        """Получение информации о торговой паре"""
        try:
            return self.client.get_symbol_info(symbol)
        except BinanceAPIException as e:
            self.logger.error(f"Ошибка получения информации о паре {symbol}: {str(e)}")
            return None

    def normalize_quantity(self, symbol, quantity):
        """Нормализация количества под требования биржи"""
        info = self.get_symbol_info(symbol)
        if not info:
            return None
            
        step_size = None
        for filter in info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                step_size = float(filter['stepSize'])
                break
                
        if step_size:
            decimals = len(str(step_size).split('.')[-1].rstrip('0'))
            return float(Decimal(str(quantity)).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN))
        return quantity

    def place_order(self, symbol, side, order_type, quantity, price=None):
        """Размещение ордера"""
        try:
            quantity = self.normalize_quantity(symbol, quantity)
            if not quantity:
                return None
                
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            if order_type == 'LIMIT':
                params['timeInForce'] = 'GTC'
                params['price'] = price
                
            order = self.client.create_order(**params)
            self.logger.info(f"Размещен ордер: {order}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Ошибка размещения ордера: {str(e)}")
            return None

    def get_balance(self, asset):
        """Получение баланса по активу"""
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance['free'])
        except BinanceAPIException as e:
            self.logger.error(f"Ошибка получения баланса {asset}: {str(e)}")
            return 0.0

    def get_open_orders(self, symbol=None):
        """Получение открытых ордеров"""
        try:
            return self.client.get_open_orders(symbol=symbol)
        except BinanceAPIException as e:
            self.logger.error(f"Ошибка получения открытых ордеров: {str(e)}")
            return []

    def cancel_order(self, symbol, order_id):
        """Отмена ордера"""
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            self.logger.error(f"Ошибка отмены ордера: {str(e)}")
            return None 