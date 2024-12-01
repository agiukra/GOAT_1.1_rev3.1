from .base_strategy import SignalStrategy
from .goat_strategy import GoatStrategy

# Словарь доступных стратегий
AVAILABLE_STRATEGIES = {
    'goat': GoatStrategy
}

def get_strategy(strategy_name, config):
    """
    Фабричный метод для создания стратегии по имени
    """
    if strategy_name not in AVAILABLE_STRATEGIES:
        raise ValueError(f"Неизвестная стратегия: {strategy_name}")
    
    strategy_class = AVAILABLE_STRATEGIES[strategy_name]
    return strategy_class(config) 