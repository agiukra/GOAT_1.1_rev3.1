
# GOAT Trader Bot с Многоагентной Архитектурой

Этот проект реализует алгоритмического торгового бота с использованием стратегии GOAT и многоагентной архитектуры. 
Система использует несколько интеллектуальных агентов, включая самообучающихся агентов, для анализа данных, 
управления рисками и выполнения сделок на бирже Binance.

## Оглавление

1. [Описание Проекта](#описание-проекта)
2. [Структура Проекта](#структура-проекта)
3. [Установка и Настройка](#установка-и-настройка)
    - [Установка Python](#установка-python)
    - [Создание виртуального окружения](#создание-виртуального-окружения)
    - [Активация виртуального окружения](#активация-виртуального-окружения)
    - [Установка зависимостей](#установка-зависимостей)
    - [Настройка Конфигурации](#настройка-конфигурации)
4. [Запуск и Тестирование Проекта](#запуск-и-тестирование-проекта)
    - [Запуск основного скрипта](#запуск-основного-скрипта)
    - [Проверка логов](#проверка-логов)
    - [Тестирование агентов](#тестирование-агентов)
5. [Рекомендации по Тестированию и Обучению](#рекомендации-по-тестированию-и-обучению)
6. [Важные Примечания](#важные-примечания)
7. [Лицензия](#лицензия)

## Описание Проекта

GOAT Trader Bot представляет собой алгоритмического торгового бота, который использует стратегию GOAT для анализа рыночных данных и выполнения сделок на бирже Binance. Бот состоит из нескольких агентов, каждый из которых отвечает за конкретную функцию — от анализа данных до управления рисками и исполнения торговых операций.

## Структура Проекта

- **agents/**: Содержит файлы агентов, отвечающих за торговлю, управление рисками, анализ данных, анализ сентимента и исполнение сделок.
- **hub/**: Центральный модуль, координирующий взаимодействие между всеми агентами.
- **config.py**: Конфигурационный файл, в котором определяются ключи API, параметры стратегии и настройки агентов.
- **data/**: Папка для хранения исторических данных для анализа и бэктестинга.
- **models/**: Папка для хранения обученных моделей для самообучающихся агентов.
- **strategies/**: Папка, содержащая реализацию торговой стратегии GOAT.
- **utils/**: Вспомогательные модули для загрузки данных и управления рисками.
- **main.py**: Основной скрипт для запуска бота.
- **requirements.txt**: Файл со списком зависимостей, необходимых для работы проекта.

## Установка и Настройка

### Установка Python

Этот проект требует Python версии **3.9**, рекомендуется использовать версию **3.9.13**. 
Скачать Python можно с [официального сайта Python](https://www.python.org/downloads/release/python-3913/).

### Создание виртуального окружения

Для управления зависимостями проекта создайте виртуальное окружение:

```bash
python -m venv goat_trader_env
```

### Активация виртуального окружения

- **На Windows**:
  ```bash
  goat_trader_env\Scripts\activate
  ```
- **На macOS/Linux**:
  ```bash
  source goat_trader_env/bin/activate
  ```

### Установка зависимостей

Установите все зависимости, указанные в файле `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Настройка Конфигурации

Откройте файл `config.py` и укажите следующие параметры:

- **API_KEY** и **API_SECRET**: Ваши ключи для доступа к API Binance.
- **BASE_URL**: URL-адрес API Binance.
- **STRATEGY_PARAMETERS**: Параметры стратегии, такие как `risk_reward_ratio`, `max_loss`, `timeframe`, и `symbol`.
- **AGENT_SETTINGS**: Настройки агентов, такие как `risk_tolerance` и `sentiment_threshold`.

Пример содержимого `config.py`:

```python
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
BASE_URL = "https://api.binance.com"

# Параметры стратегии
STRATEGY_PARAMETERS = {
    "risk_reward_ratio": 3,
    "max_loss": 0.01,
    "timeframe": "1h",
    "symbol": "BTCUSDT",
}

# Настройки агентов
AGENT_SETTINGS = {
    "risk_tolerance": 0.01,
    "sentiment_threshold": 0.5,
}
```

## Запуск и Тестирование Проекта

### Запуск основного скрипта

Для запуска проекта выполните основной скрипт `main.py`:

```bash
python main.py
```

### Проверка логов

Во время работы бота будут выводиться логи, которые содержат информацию о действиях агентов:

- **TradingAgent**: Генерирует торговые сигналы на основе стратегии GOAT.
- **RiskAgent**: Рассчитывает размеры позиций и управляет рисками.
- **DataAgent**: Загружает исторические данные и данные в реальном времени.
- **SentimentAgent**: Анализирует новости и социальные сети для определения сентимента.
- **ExecutorAgent**: Исполняет торговые ордера на Binance.

### Тестирование агентов

Перед запуском на реальном счёте рекомендуется протестировать каждого агента:

- **TradingAgent**: Проверьте, как агент генерирует сигналы в зависимости от исторических данных.
- **RiskAgent**: Оцените, как рассчитываются уровни риска и размеры позиций.
- **DataAgent**: Убедитесь, что данные загружаются корректно.
- **SentimentAgent**: Проверьте, корректно ли анализируются новости и определяется сентимент.
- **ExecutorAgent**: Тестируйте выполнение сделок на тестовом аккаунте или с минимальными средствами.

## Рекомендации по Тестированию и Обучению

1. **Эмуляция и Бэктестинг**: Используйте исторические данные для тестирования бота и оценки его поведения в различных рыночных условиях.
2. **Переобучение моделей**: Регулярно переобучайте самообучающихся агентов, чтобы они могли адаптироваться к текущим условиям рынка.
3. **Мониторинг и Логирование**: Включите логирование для отслеживания работы агентов, особенно во время торговли на реальных средствах.

## Важные Примечания

- **Реальная Торговля**: Перед запуском на реальных средствах протестируйте систему на тестовом аккаунте Binance или с небольшим балансом.
- **Контроль Рисков**: Убедитесь, что настройки агентов по управлению рисками настроены на минимизацию потерь.
- **Обновления**: Регулярно обновляйте библиотеки и переобучайте модели для поддержания актуальности бота.

## Лицензия

Этот проект предоставляется в образовательных целях и не предназначен для использования в реальной торговле без тщательного тестирования.