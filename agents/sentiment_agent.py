from transformers import pipeline

class SentimentAgent:
    def __init__(self):
        # Используем NLP-модель для анализа сентимента
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def analyze_sentiment(self, news_text):
        # Анализ сентимента текста и возврат результата
        result = self.sentiment_pipeline(news_text)
        score = result[0]["score"] if result[0]["label"] == "POSITIVE" else -result[0]["score"]
        return score
