def calculate_position_size(balance, risk_percent, stop_loss_distance):
    risk_amount = balance * risk_percent
    return risk_amount / stop_loss_distance

def apply_risk_management(balance, entry_price, stop_loss_price):
    stop_loss_distance = entry_price - stop_loss_price
    return calculate_position_size(balance, 0.01, stop_loss_distance)

def calculate_max_position_size(self):
    """Расчет максимального размера позиции"""
    try:
        # Получаем баланс и параметры из конфига
        balance = self.get_balance()
        max_position_percent = self.config.get('max_position_percent', 5)
        
        # Рассчитываем максимальный размер в USDT
        max_position_size = balance * (max_position_percent / 100)
        
        # Применяем дополнительные ограничения
        min_position = self.config.get('min_position_size', 10)
        absolute_max = self.config.get('absolute_max_position', 1000)
        
        # Проверяем ограничения
        if max_position_size < min_position:
            self.logger.warning(f"Максимальный размер позиции ({max_position_size:.2f}) меньше минимального ({min_position})")
            return 0
            
        return min(max_position_size, absolute_max)
        
    except Exception as e:
        self.logger.error(f"Ошибка расчета максимального размера позиции: {str(e)}")
        return 0

def validate_position_size(self, size, price):
    """Проверка размера позиции"""
    try:
        position_value = size * price
        max_position_size = self.calculate_max_position_size()
        
        if position_value > max_position_size:
            self.logger.warning(
                f"Размер позиции {position_value:.2f} USDT превышает максимально допустимый {max_position_size:.2f} USDT"
            )
            return False
            
        return True
        
    except Exception as e:
        self.logger.error(f"Ошибка валидации размера позиции: {str(e)}")
        return False
