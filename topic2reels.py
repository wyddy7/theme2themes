#!/usr/bin/env python
"""Topic2Reels — CLI-утилита для генерации субтопиков, JSON и файлов Markdown по заданной теме с помощью OpenAI-совместимой LLM.

Запуск:
    python topic2reels.py "<ТЕМА>" [--config path/to/theme-config.yaml] [--output_dir resources/] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

import yaml  # type: ignore
from dotenv import load_dotenv  # type: ignore
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table
from rich import box
from slugify import slugify  # type: ignore
import openai  # type: ignore

# ----------------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------------

class Subtopic(BaseModel):
    title: str
    context: str
    order: int


class TopicData(BaseModel):
    topic: str
    subtopics: List[Subtopic]


# ----------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------

console = Console()

# --------------------------- Logging setup ---------------------------------

def setup_logging(cfg: Dict[str, Any]) -> Path:
    log_cfg = cfg.get("logging", {})
    level_console = getattr(logging, log_cfg.get("level_console", "INFO"))
    level_file = getattr(logging, log_cfg.get("level_file", "DEBUG"))
    use_json = bool(log_cfg.get("json", False))
    max_bytes = int(log_cfg.get("max_bytes", 1_048_576))
    backup_count = int(log_cfg.get("backup_count", 3))

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"topic-{int(time.time())}.log"

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # File handler (rotation)
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    if use_json:
        formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level_file)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(level_console)
    logger.addHandler(console_handler)

    # Silence httpx / openai noise unless debugging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    console.print(f"[bold]Лог пишется в файл: {log_file}")
    return log_file


logger = logging.getLogger(__name__)

# Global counters for summary
stats = {"calls": 0, "retries": 0, "backoff_sec": 0.0}

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        console.print(f"[red]Config file not found: {path}")
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_api() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Environment variable OPENAI_API_KEY is not set.\nCreate a .env file or export it.")
        sys.exit(1)
    openai.api_key = api_key
    base_url = os.getenv("PROXY_URL")
    if base_url:
        openai.base_url = base_url  # type: ignore


def price_usd(model: str, usage: Any, price_table: Dict[str, Any]) -> float:
    """Считает стоимость запроса, поддерживая как старую (flat) схему,
    так и новую: {'model': {'in': 0.003, 'out': 0.015}}."""
    if usage is None:
        return 0.0

    # новая схема — словарь с in/out тарифами
    entry = price_table.get(model)
    if isinstance(entry, dict):
        in_rate = float(entry.get("in", 0.0))
        out_rate = float(entry.get("out", 0.0))
        prompt_cost = (usage.prompt_tokens / 1000) * in_rate
        completion_cost = (usage.completion_tokens / 1000) * out_rate
        return prompt_cost + completion_cost

    # старая схема — единая цена за 1k total tokens
    if entry is not None:
        return (usage.total_tokens / 1000) * float(entry)

    return 0.0


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def step_cfg(cfg: Dict[str, Any], name: str) -> Any:
    """Возвращает конфиг конкретного шага (dict или bool)."""
    return cfg.get("steps", {}).get(name, True)


def step_enabled(cfg: Dict[str, Any], name: str) -> bool:
    s = step_cfg(cfg, name)
    if isinstance(s, dict):
        return s.get("enabled", True)
    return bool(s)


def step_model(cfg: Dict[str, Any], name: str, default_model_key: str) -> str:
    s = step_cfg(cfg, name)
    if isinstance(s, dict) and s.get("model"):
        return s["model"]
    return cfg.get(default_model_key)


# ----------------------------------------------------------------------------
# Processing steps
# ----------------------------------------------------------------------------


def step_generate(topic: str, cfg: Dict[str, Any]) -> str:
    template: str = cfg["prompts"]["generate"]
    prompt = template.format(topic=topic, max_subtopics=cfg.get("max_subtopics", 12))
    messages = [{"role": "user", "content": prompt}]
    model = step_model(cfg, "generate", "model_primary")
    console.print(f"[cyan]→ (generate) Модель: {model}")
    resp = call_chat("generate", model, messages, cfg)
    usage = resp.usage  # type: ignore
    cost = price_usd(model, usage, cfg.get("price_per_1k_tokens", {})) if usage else 0.0
    console.print(f"[green]   ✓ Получено. Токены: {usage.total_tokens if usage else 'n/a'} | Цена: ${cost:.4f}")
    content = str(resp.choices[0].message.content).strip()
    console.print(f"[blue]Ответ модели (to_json):\n{content[:1000]}{'...' if len(content)>1000 else ''}")
    return content, cost


def step_to_json(raw_list: str, topic: str, cfg: Dict[str, Any]) -> TopicData:
    template: str = cfg["prompts"]["to_json"]
    prompt = template + "\n\nСписок подтем:\n" + raw_list
    messages = [{"role": "user", "content": prompt}]
    model = cfg["model_secondary"]
    console.print(f"[cyan]→ Преобразую в JSON через {model}…")
    resp = call_chat("to_json", model, messages, cfg)
    usage = resp.usage  # type: ignore
    cost = price_usd(model, usage, cfg.get("price_per_1k_tokens", {})) if usage else 0.0
    console.print(f"[green]   ✓ JSON получен. Токены: {usage.total_tokens if usage else 'n/a'} | Цена: ${cost:.4f}")
    content = str(resp.choices[0].message.content).strip()
    content_clean = strip_code_fences(content)
    logger.info("Cleaned JSON content: %s", content_clean)
    try:
        data = json.loads(content_clean)
    except json.JSONDecodeError as e:
        console.print(f"[red]Не удалось распарсить JSON: {e}\nОтвет модели:\n{content}")
        logger.error("JSONDecodeError: %s", e)
        sys.exit(1)
    try:
        topic_data = TopicData(**data)
    except ValidationError as v:
        console.print(f"[red]JSON не соответствует схеме pydantic:\n{v}")
        sys.exit(1)
    return topic_data, cost


def strip_code_fences(text: str) -> str:
    """Удаляет Markdown-блоки ``` и ```json, если они присутствуют."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # убираем первую строчку с ``` или ```json
        lines = lines[1:]
        # убираем последнюю, если она ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def step_expand(topic_data: TopicData, cfg: Dict[str, Any]) -> tuple[TopicData, float]:
    """Расширяет контексты каждой подтемы. Возвращает обновлённый TopicData и добавленную стоимость."""
    template: str = cfg["prompts"]["expand"]
    model = step_model(cfg, "expand", "model_secondary")
    total_cost = 0.0
    for st in topic_data.subtopics:
        prompt = template.format(subtopic_title=st.title, topic=topic_data.topic)
        messages = [{"role": "user", "content": prompt}]
        console.print(f"[cyan]→ (expand) '{st.title}' | модель: {model}")
        resp = call_chat("expand", model, messages, cfg)
        usage = resp.usage  # type: ignore
        cost = price_usd(model, usage, cfg.get("price_per_1k_tokens", {})) if usage else 0.0
        console.print(f"[green]   ✓ Готово. Токены: {usage.total_tokens if usage else 'n/a'} | Цена: ${cost:.4f}")
        total_cost += cost
        st.context = str(resp.choices[0].message.content).strip()
        console.print(f"[blue]Ответ модели (expand) для '{st.title}':\n{st.context[:500]}{'...' if len(st.context)>500 else ''}")
    return topic_data, total_cost


# ----------------------------------------------------------------------------
# File operations
# ----------------------------------------------------------------------------

def save_outputs(topic_data: TopicData, output_dir: Path, cfg: Dict[str, Any]) -> None:
    """Сохраняет результаты.

    Режимы:
      1. split  (legacy) — каталог + master.json + отдельные .md
      2. json   — один <slug>.json, внутри полный контент подтем
    """

    slug = make_slug(topic_data.topic, cfg)
    mode = cfg.get("output_mode", "split").lower()

    if mode == "json":
        # Единый human-/machine-readable файл
        output_path = output_dir / f"{slug}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(topic_data.model_dump(), f, ensure_ascii=False, indent=2)
        console.print(f"[green]✓ JSON сохранён: {output_path}")
        return

    # === legacy 'split' режим ===
    topic_dir = output_dir / slug
    topic_dir.mkdir(parents=True, exist_ok=True)

    # master.json
    master_path = topic_dir / "master.json"
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(topic_data.model_dump(), f, ensure_ascii=False, indent=2)

    # subtopics md files
    for st in topic_data.subtopics:
        file_name = f"{st.order:02d}_{make_slug(st.title, cfg)}.md"
        path = topic_dir / file_name
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# {st.title}\n\n{st.context}\n")

    console.print(f"[green]✓ Файлы сохранены в {topic_dir}")


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def report(topic_data: TopicData, total_cost: float) -> None:
    table = Table(title=f"Subtopics for '{topic_data.topic}'", box=box.SIMPLE)
    table.add_column("#")
    table.add_column("Title")
    for st in topic_data.subtopics:
        table.add_row(str(st.order), st.title)
    console.print(table)
    console.print(f"[bold]Общая стоимость запросов: ${total_cost:.4f}")


# ----------------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtopics and content files for a given topic via LLM.")
    parser.add_argument("topic", help="Тема, которую нужно разбить на подтемы")
    parser.add_argument("--config", default="theme-config.yaml", help="Путь к YAML-конфигу")
    parser.add_argument("--output_dir", default="resources", help="Каталог для сохранения файлов")
    parser.add_argument("--source_file", help="Путь к файлу с уже готовым контекстом (обойдёт шаг generate)")
    parser.add_argument("--dry_run", action="store_true", help="Не сохранять файлы")
    parser.add_argument("--quiet", action="store_true", help="Минимум вывода в консоль")
    parser.add_argument("--debug", action="store_true", help="Включить расширенное логирование в консоль")
    args = parser.parse_args()

    if args.quiet:
        console.quiet = True  # type: ignore

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[yellow]DEBUG режим активирован")

    ensure_api()
    cfg = load_config(Path(args.config))
    log_path = setup_logging(cfg)

    total_cost = 0.0

    topic_data: TopicData | None = None

    if args.source_file:
        # ---------- Путь с готовым текстовым контекстом ----------
        src_path = Path(args.source_file)
        if not src_path.exists():
            console.print(f"[red]Source file not found: {src_path}")
            sys.exit(1)
        raw_context = src_path.read_text(encoding="utf-8")
        console.print(f"[cyan]→ Обрабатываю контекст из {src_path}…")
        topic_data, cost = step_from_context(raw_context, args.topic, cfg)
        total_cost += cost
    else:
        # ---------- Автопоиск контекста в input/<slug>.txt|.md ----------
        input_dir = Path(cfg.get("input_dir", "input"))
        slug = make_slug(args.topic, cfg)
        found_file = None
        for ext in (".txt", ".md"):
            candidate = input_dir / f"{slug}{ext}"
            if candidate.exists():
                found_file = candidate
                break

        if found_file is not None:
            raw_context = found_file.read_text(encoding="utf-8")
            console.print(f"[cyan]→ Нашёл входной файл {found_file}. Использую его вместо шага generate.")
            topic_data, cost = step_from_context(raw_context, args.topic, cfg)
            total_cost += cost
        else:
            # ---------- Обычный генеративный путь ----------
            raw_list = None
            if step_enabled(cfg, "generate"):
                raw_list, cost = step_generate(args.topic, cfg)
                total_cost += cost
            else:
                console.print("[yellow]Шаг generate пропущен конфигом, требуется кэш.")
                sys.exit(1)

            if step_enabled(cfg, "to_json"):
                topic_data, cost = step_to_json(raw_list, args.topic, cfg)
                total_cost += cost
            else:
                console.print("[yellow]Шаг to_json пропущен, требуется кэш.")
                sys.exit(1)

    # Cost limit check
    cost_limit = cfg.get("cost_limit_usd", 0.05)
    if total_cost > cost_limit:
        console.print(f"[red]Превышен лимит стоимости ${cost_limit:.2f} (факт {total_cost:.4f}). Прерываю.")
        sys.exit(1)

    # Step 3: expand
    if step_enabled(cfg, "expand"):
        topic_data, cost = step_expand(topic_data, cfg)
        total_cost += cost

    # Step 4: save files
    if step_enabled(cfg, "save_files") and not args.dry_run:
        save_outputs(topic_data, Path(args.output_dir), cfg)
    else:
        console.print("[yellow]Сохранение файлов пропущено (dry_run или отключено).")

    # Step 5: report
    if step_enabled(cfg, "report"):
        report(topic_data, total_cost)

    # summary at end
    console.print(f"[bold]=== RUN SUMMARY ===")
    console.print(f"Запросов: {stats['calls']} | Повторов: {stats['retries']} | Время ожидания: {stats['backoff_sec']:.1f} c")
    logger.info("SUMMARY calls=%s retries=%s backoff_sec=%.1f", stats['calls'], stats['retries'], stats['backoff_sec'])


def call_chat(step: str, model: str, messages: List[Dict[str, str]], cfg: Dict[str, Any]) -> Any:
    retry_cfg = cfg.get("retry", {})
    max_tries = int(retry_cfg.get("max_tries", 5))
    base = float(retry_cfg.get("backoff_base", 0.5))
    factor = float(retry_cfg.get("backoff_factor", 2))

    for attempt in range(1, max_tries + 1):
        try:
            stats["calls"] += 1
            logger.info("STEP %s | MODEL %s | try %s/%s | prompt=%s", step, model, attempt, max_tries, messages[0]["content"][:200].replace("\n", " "))
            response = openai.chat.completions.create(model=model, messages=messages)  # type: ignore
            logger.info("STEP %s | MODEL %s | response=%s", step, model, response.choices[0].message.content[:500].replace("\n", " "))
            if response.usage:
                logger.info("STEP %s | MODEL %s | tokens=%s", step, model, response.usage.total_tokens)
            return response
        except (openai.RateLimitError, openai.APIError, openai.APIConnectionError) as e:
            stats["retries"] += 1
            logger.warning("STEP %s | MODEL %s | attempt %s failed: %s", step, model, attempt, str(e)[:300])
            if attempt == max_tries:
                logger.error("STEP %s | MODEL %s | exhausted retries", step, model)
                raise
            sleep_sec = base * (factor ** (attempt - 1)) * (1 + random.random() * 0.3)
            stats["backoff_sec"] += sleep_sec
            console.print(f"[yellow]Rate limited / error. Повтор через {sleep_sec:.1f} с… (попытка {attempt}/{max_tries})")
            time.sleep(sleep_sec)


# ----------------------------------------------------------------------------
# Slug helper
# ----------------------------------------------------------------------------

def make_slug(text: str, cfg: Dict[str, Any]) -> str:
    file_cfg = cfg.get("file_naming", {})
    allow_unicode = bool(file_cfg.get("allow_unicode", False))
    return slugify(text, separator="-", allow_unicode=allow_unicode)


# ----------------------------------------------------------------------------
# New step: build JSON from raw context text
# ----------------------------------------------------------------------------


def step_from_context(raw_text: str, topic: str, cfg: Dict[str, Any]) -> tuple[TopicData, float]:
    """Преобразует предоставленный пользователем контекст в TopicData через LLM."""
    template: str = cfg["prompts"].get("from_context") or cfg["prompts"]["to_json"]
    prompt = template.format(topic=topic, context=raw_text, max_subtopics=cfg.get("max_subtopics", 12))
    messages = [{"role": "user", "content": prompt}]
    model = cfg.get("model_secondary")
    console.print(f"[cyan]→ (from_context) Модель: {model}")
    resp = call_chat("from_context", model, messages, cfg)
    usage = resp.usage  # type: ignore
    cost = price_usd(model, usage, cfg.get("price_per_1k_tokens", {})) if usage else 0.0
    console.print(f"[green]   ✓ JSON получен. Токены: {usage.total_tokens if usage else 'n/a'} | Цена: ${cost:.4f}")

    content = str(resp.choices[0].message.content).strip()
    content_clean = strip_code_fences(content)
    try:
        data = json.loads(content_clean)
    except json.JSONDecodeError as e:
        console.print(f"[red]Не удалось распарсить JSON: {e}\nОтвет модели:\n{content}")
        sys.exit(1)
    topic_data = TopicData(**data)
    return topic_data, cost


if __name__ == "__main__":
    main() 