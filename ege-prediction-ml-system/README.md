# EGE Prediction ML System

ML-система прогнозирования баллов ЕГЭ на основе данных онлайн-школы. Использует CatBoost для регрессии, FastAPI для API-инференса, Streamlit для UI, PostgreSQL для хранения данных и MLflow для трекинга экспериментов.

## Архитектура

```
ege-prediction-ml-system/
├── app/
│   ├── api/              # FastAPI — REST-эндпоинты для предсказания
│   ├── batch/            # Пакетное предсказание из PostgreSQL
│   ├── core/             # Константы, пути, конфигурация фичей
│   ├── db/               # PostgreSQL репозиторий и настройки
│   ├── frontend/         # Streamlit UI
│   └── training/         # Подготовка данных, обучение, MLflow
├── artifacts/            # Сохранённая модель и метрики
├── data/                 # CSV-файлы с данными
├── mlflow/               # Локальное хранилище MLflow
├── scripts/              # Shell-скрипты для запуска
├── sql/                  # DDL для инициализации таблиц
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Стек технологий

- **Модель**: CatBoostRegressor (MAE ~ 5.93, R2 ~ 0.63)
- **API**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **БД**: PostgreSQL 16
- **Эксперименты**: MLflow
- **Контейнеризация**: Docker Compose

## Быстрый старт (Docker)

### 1. Подготовка данных

Положите файл `2025_07_24_el_school.csv` в папку `data/`.

### 2. Запуск инфраструктуры

```bash
docker-compose up -d
```

Поднимутся PostgreSQL, MLflow, FastAPI и Streamlit.

### 3. Обучение модели

```bash
docker-compose --profile train run --rm training
```

### 4. Batch scheduler (опционально)

```bash
docker-compose --profile batch up -d batch-scheduler
```

### Доступные сервисы

| Сервис | URL |
|---|---|
| API | http://localhost:8000 |
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| PostgreSQL | localhost:5432 |

## Локальный запуск (без Docker)

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Обучение

```bash
python -m app.training.train_with_mlflow --tracking-uri http://localhost:5000 --register-model
```

Без MLflow:

```bash
python -m app.training.train_with_mlflow --tracking-uri ""
```

### API

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend

```bash
streamlit run app/frontend/streamlit_app.py
```

### Batch prediction

Одиночный запуск:

```bash
python -m app.batch.run_batch_prediction --limit 100 --model-version local-catboost-v1
```

Непрерывный планировщик (каждые 5 минут):

```bash
BATCH_INTERVAL_SECONDS=300 python -m app.batch.scheduler
```

## API Endpoints

### GET /health

Проверка состояния сервиса.

### POST /predict

Прогноз балла ЕГЭ по данным студента.

Пример запроса:

```json
{
  "student_target": "90.0",
  "student_class": "11",
  "course_name": "Годовой 2к25 стандарт",
  "subject_name": "Обществознание",
  "homework_done_mark": 70.0,
  "test_part_one": 60.0,
  "test_part_two": 40.0
}
```

Ответ:

```json
{
  "predicted_ege_score": 72.45
}
```

## Пайплайн данных

1. **Очистка**: фильтрация по валидным баллам ЕГЭ (30-100), классам (9-11), целевому баллу (0-100), разделение тестов на часть 1 и часть 2
2. **Feature engineering**: лаги (1, 2 урока назад), разности, скользящие средние и стандартные отклонения за 3 урока, экстремумы по студенту
3. **Агрегация**: понедельная группировка по студенту, предмету, типу курса
4. **Удаление выбросов**: IQR-метод с коэффициентом 2

## Переменные окружения

| Переменная | Значение по умолчанию | Описание |
|---|---|---|
| POSTGRES_HOST | localhost | Хост PostgreSQL |
| POSTGRES_PORT | 5432 | Порт PostgreSQL |
| POSTGRES_DB | ege_predictions | Имя базы данных |
| POSTGRES_USER | postgres | Пользователь БД |
| POSTGRES_PASSWORD | postgres | Пароль БД |
| MLFLOW_TRACKING_URI | http://localhost:5000 | URI MLflow сервера |
| API_BASE_URL | http://localhost:8000 | URL FastAPI для фронтенда |
| BATCH_INTERVAL_SECONDS | 300 | Интервал batch scheduler (сек) |
| BATCH_LIMIT | 100 | Размер батча |
