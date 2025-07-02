# Техническое задание: Topic2Reels

## 1. Общие сведения
* **Название проекта:** Topic2Reels
* **Язык реализации:** Python ≥ 3.10
* **Способ запуска:** CLI-утилита (один файл `topic2reels.py`)
* **Библиотеки:** `openai` (или совместимый LiteLLM-proxy), `pydantic`, `python-dotenv`, `tqdm`, `rich`
* **Файлы конфигурации**  
  * `.env` — секреты (`OPENAI_API_KEY`, `PROXY_URL`, …)  
  * `theme-config.yaml` — рабочие параметры (модели, промпты, выключатели шагов)

## 2. Функциональные требования
### 2.1 Ввод
* Запуск:
  ```bash
  python topic2reels.py "<ТЕМА>" [--config path/to/theme-config.yaml]
  ```
* **Обязательный аргумент:** текстовая тема.
* **Опциональные параметры:**
  * `--output_dir` (по умолчанию `resources/`)
  * `--verbose` / `--quiet`
  * `--dry_run` — не записывать файлы

### 2.2 Обработка
**Шаг 1. Генерация субтопиков**  
* Модель A (напр. `gpt-4o`)  
* Промпт из config создаёт список *N* (≤ 20) подтем + краткий контекст (2-3 предложения).

**Шаг 2. Преобразование в JSON**  
* Модель B (бюджетная, напр. `gpt-3.5-turbo`)  
* Промпт возвращает строгий JSON:
  ```json
  {
    "topic": "…",
    "subtopics": [
      { "title": "…", "context": "…", "order": 1 }
    ]
  }
  ```

**Шаг 3. Сохранение результатов**  
* Создаётся каталог `resources/<slug-от-темы>/`
* `master.json` — полный JSON из шага 2
* Для каждой подтемы:
  * `resources/<slug>/<order>_<slug>.md`
  * Содержимое:
    ```markdown
    # <TITLE>
    <CONTEXT>
    ```

**Шаг 4. Отчёт**  
* В консоль — таблица subtopic → файл
* Логирование в `logs/topic-<timestamp>.log`

### 2.3 Выключение шагов
Каждый шаг имеет флаг в `theme-config.yaml`: `generate`, `to_json`, `save_files`, `report`. Пропущенный шаг берёт кэш предыдущего запуска, если есть.

## 3. Нефункциональные требования
* Один исполняемый файл (`topic2reels.py`), но код структурирован функциями
* Типизация (PEP 484) и валидация JSON через `pydantic`
* Обработка ошибок API, повтор при rate-limit (эксп. backoff)
* Лимит стоимости: ≤ $0.05 за запуск (настраивается)
* Логи цветные и понятные (`rich`)
* Кроссплатформенность (Win/macOS/Linux)
* Время выполнения ≤ 30 сек при *N* ≤ 10 подтем

## 4. Структура репозитория
```
resources/                     # Автовыход
scripts/validate_json.py       # Вспом. проверка
topic2reels.py                 # Главный скрипт
theme-config.yaml              # Пример конфигурации
.env.example                   # Шаблон секретов
README.md                      # Инструкция запуска
```

## 5. Пример `theme-config.yaml`
```yaml
model_primary: "gpt-4o-mini"        # этап 1
model_secondary: "gpt-3.5-turbo"    # этап 2
max_subtopics: 12
prompts:
  generate: |
    Сформируй …
  to_json: |
    Преобразуй список …
steps:
  generate: true
  to_json: true
  save_files: true
  report: true
```

## 6. Критерии приёмки
* Проект ставится `pip install -r requirements.txt`, запускается без ошибок
* Тема «Финансовая грамотность» создаёт `resources/finansovaya-gramotnost` с `master.json` и MD-файлами
* `master.json` валиден по схеме `pydantic`
* При `--dry_run` файлы не создаются
* Стоимость ≤ лимита

## 7. Дополнительные задачи (опционально)
* Юнит-тесты на парсинг config и сохранение файлов
* Метка времени и UUID в `master.json`
* Асинхронный вызов дешёвой модели (asyncio) 