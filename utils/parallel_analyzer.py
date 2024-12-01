from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

class ParallelPairAnalyzer:
    def __init__(self, analyzer, max_workers=10):
        self.analyzer = analyzer
        self.max_workers = max_workers
        self.logger = logging.getLogger('TradingBot.ParallelAnalyzer')

    def analyze_pairs_batch(self, pairs, batch_size=50):
        """Анализ пар пакетами для снижения нагрузки на API"""
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_results = self.analyze_pairs_parallel(batch)
            results.extend(batch_results)
            self.logger.info(f"Проанализировано {min(i + batch_size, len(pairs))}/{len(pairs)} пар")
        return results

    def analyze_pairs_parallel(self, pairs):
        """Параллельный анализ пар"""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pair = {
                executor.submit(self.safe_analyze_pair, pair): pair 
                for pair in pairs
            }
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Ошибка анализа пары {pair}: {str(e)}")
        return results

    def safe_analyze_pair(self, pair):
        """Безопасный анализ одной пары"""
        try:
            return self.analyzer.analyze_pair(pair)
        except Exception as e:
            self.logger.error(f"Ошибка анализа пары {pair}: {str(e)}")
            return None 