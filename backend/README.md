# EGE Math Solution Checker - Backend

Бэкенд для системы автоматической проверки решений задач ЕГЭ по математике с использованием моделей рассуждения через OpenRouter API.

## Требования

- Python 3.8+
- FastAPI
- OpenRouter API ключ

## Установка

1. Клонируйте репозиторий:

```bash
git clone <repository-url>
cd <repository-folder>/backend
```

2. Создайте и активируйте виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` на основе `.env.example` и добавьте свой OpenRouter API ключ:

```bash
cp .env.example .env
# Отредактируйте файл .env, добавив свой API ключ
```

## Запуск

Запустите сервер разработки:

```bash
uvicorn app.main:app --reload
```

Сервер будет доступен по адресу http://localhost:8000.

## API Endpoints

### Проверка решений

- `POST /api/v1/solutions/evaluate` - Оценить решение задачи

### Информация о моделях

- `GET /api/v1/models/available` - Получить список доступных моделей
  - Параметр `category` - фильтр по категории моделей (`reasoning` или `non_reasoning`)
- `GET /api/v1/models/task-types` - Получить список типов задач
- `GET /api/v1/models/openrouter-models` - Получить список всех моделей из OpenRouter
- `GET /api/v1/models/credits` - Получить информацию о кредитах аккаунта
- `GET /api/v1/models/cost-estimate/{model_id}` - Получить оценку стоимости использования модели для всего датасета
- `GET /api/v1/models/compare-costs` - Сравнить стоимость использования разных моделей

## Тестирование

Запустите тесты с помощью pytest:

```bash
pytest
```

## Структура проекта

```
backend/
├── app/
│   ├── api/
│   │   └── openrouter_client.py  # Клиент для работы с OpenRouter API
│   ├── core/
│   │   └── config.py             # Конфигурация приложения
│   ├── models/
│   │   └── solution.py           # Модели данных
│   ├── routers/
│   │   ├── models.py             # Маршруты для работы с моделями
│   │   └── solutions.py          # Маршруты для проверки решений
│   ├── utils/
│   │   ├── image_utils.py        # Утилиты для работы с изображениями
│   │   ├── prompt_utils.py       # Утилиты для формирования промптов
│   │   └── cost_calculator.py    # Утилиты для расчета стоимости
│   └── main.py                   # Основной файл приложения
├── tests/
│   └── test_openrouter_client.py # Тесты для клиента OpenRouter
├── .env.example                  # Пример файла с переменными окружения
├── requirements.txt              # Зависимости проекта
└── README.md                     # Документация
```
