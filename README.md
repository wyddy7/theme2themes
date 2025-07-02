# Topic2Reels

CLI-утилита для генерации подтем и контента на основе LLM.

## Установка зависимостей
```bash
pip install -r requirements.txt
```

## Настройка переменных окружения
Создайте файл `.env` в корне и добавьте:
```
OPENAI_API_KEY="sk-…"
# Необязательно, если используете прокси LiteLLM
PROXY_URL="https://your-litellm-proxy.example.com/v1"
```

## Пример запуска
```bash
python topic2reels.py "Финансовая грамотность"
```
Опции:
* `--config path/to/theme-config.yaml` — альтернативный конфиг
* `--output_dir data/` — куда сохранять файлы
* `--dry_run` — не сохранять файлы
* `--quiet` — минимальный вывод
* `--source_file path/to/context.txt` — взять готовый текстовый конспект, пропустить генерацию

> Если `--source_file` не указан, скрипт автоматически ищет файл `input/<slug>.txt` или `.md`,
> где `<slug>` — слаг темы (например, `линейная-регрессия.txt`). При наличии такого файла
> он используется вместо шага *generate*.

## Быстрый пример с собственным контекстом
```bash
# положите текстовый файл в input/линейная-регрессия.txt
python topic2reels.py "Линейная регрессия"
# или явно
python topic2reels.py "Линейная регрессия" --source_file input/lin_reg.txt
```

## Режимы сохранения результата
Настраивается в `theme-config.yaml` параметром `output_mode`:

* `json`  — один файл `resources/<slug>.json` (рекомендуется)
* `split` — каталог + `master.json` + отдельные `.md` (legacy)

## Структура выхода
### output_mode: json
```
resources/<slug>.json
```

### output_mode: split
```
resources/<slug>/
├─ master.json
├─ 01_<subtopic>.md
├─ 02_<subtopic>.md
└─ …
``` 