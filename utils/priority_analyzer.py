import heapq
import logging
from datetime import datetime
from queue import PriorityQueue
from threading import Thread

class PriorityPairAnalyzer:
    def __init__(self, client, num_workers=5):
        self.client = client
        self.logger = logging.getLogger('TradingBot.PriorityAnalyzer')
        self.task_queue = PriorityQueue()
        self.result_queue = PriorityQueue()
        self.workers = []
        self.num_workers = num_workers

    def start_workers(self):
        """Запуск worker потоков"""
        for _ in range(self.num_workers):
            worker = Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def analyze_pairs(self, pairs):
        """Анализ пар с приоритетами"""
        # Определяем приоритеты
        priorities = self._calculate_priorities(pairs)
        
        # Добавляем задачи в очередь
        for pair in pairs:
            priority = priorities.get(pair, 0)
            self.task_queue.put((priority, pair))

        # Ждем результаты
        results = []
        while len(results) < len(pairs):
            priority, result = self.result_queue.get()
            if result:
                results.append(result)

        return sorted(results, key=lambda x: x['score'], reverse=True)

    def _calculate_priorities(self, pairs):
        """Расчет приоритетов для пар"""
        priorities = {}
        for pair in pairs:
            # Приоритет на основе объема и волатильности
            try:
                ticker = self.client.get_ticker(symbol=pair)
                volume = float(ticker['volume']) * float(ticker['lastPrice'])
                volatility = abs(float(ticker['priceChangePercent']))
                
                # Более высокий приоритет для пар с большим объемом и волатильностью
                priority = (volume * volatility) / 1000000
                priorities[pair] = -priority  # Отрицательный для heapq (меньше = выше приоритет)
            except:
                priorities[pair] = 0
        return priorities 