from abc import ABC, abstractmethod

class SignalStrategy(ABC):
    """Базовый класс для торговых стратегий"""
    
    def __init__(self, symbol: str, timeframe: str):
        """Инициализация базового класса стратегии"""
        self.symbol = symbol
        self.timeframe = timeframe
    
    @abstractmethod
    def generate_signal(self, data, indicators):
        """
        Генерация торгового сигнала
        
        Args:
            data (pd.DataFrame): Исторические данные
            indicators (dict): Рассчитанные индикаторы
            
        Returns:
            str: Сигнал ('BUY', 'SELL', 'HOLD')
        """
        pass
    
    @abstractmethod
    def get_name(self):
        """Получение названия стратегии"""
        pass
    
    @abstractmethod
    def get_description(self):
        """Получение описания стратегии"""
        pass
    
    @abstractmethod
    def get_parameters(self):
        """Получение параметров стратегии"""
        pass

    @abstractmethod
    def validate_data(self, data):
        """Проверка наличия необходимых данных"""
        pass