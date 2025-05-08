# -*- coding: utf-8 -*-

import json
import os
import traceback
import sys
import time
import requests
import re

# --- Конфигурация ---
# Путь к ВХОДНОМУ файлу части с описаниями (например, part_3080ti_1.jsonl)
# ЭТОТ ПУТЬ НЕОБХОДИМО БУДЕТ СКОРРЕКТИРОВАТЬ НА КАЖДОЙ МАШИНЕ!
input_jsonl_path = "./wiki_seeds_with_descriptions/part_3080.jsonl"

# Папка для сохранения сгенерированных полных статей
output_directory = "./generated_wiki_articles"

# Шаблон имени выходного файла. {machine_name} будет заменен на основе имени входного файла части.
output_filename_pattern = "generated_wiki_articles_{machine_name}.jsonl"

# --- Параметры LMStudio API ---
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions" # Уточните в LMStudio
MODEL_NAME_IN_LMSTUDIO = "qwen3-8b" # Уточните точное имя вашей модели в LMStudio

# Параметры генерации статьи
# Temperature устанавливается глобально в LMStudio
MAX_TOKENS_ARTICLE = 2500 # Безопасный лимит токенов для статьи (больше 7000 символов)

# Задержка между запросами к API (в секундах)
RATE_LIMIT_DELAY = 0.1

# Флаг, указывающий, нужно ли пропускать записи с description_status != 'ok'
SKIP_UNCLEAR_SEEDS = True # Рекомендуется True

# Регулярное выражение для удаления тегов <think>...</think> из ответа модели
THINK_TAG_REGEX = re.compile(r'<think>.*?</think>', re.DOTALL)
# --- Конец Конфигурации ---


# --- Промпт для модели (только заголовок и описание) ---
# {title}, {description} будут заменены реальными данными
PROMPT_TEMPLATE_ARTICLE = """Ты — эксперт по написанию научно-популярных статей для энциклопеции.
Твоя задача: написать связный текст для энциклопедической статьи на русском языке, основанную на предоставленных заголовке и кратком описании.

<title>{title}</title>
<description>{description}</description>

Объём — 5–6 абзацев.
Текст должен быть полностью связанным, без списков, без подзаголовков, без маркированных или нумерованных пунктов, без ссылок на источники, без примечаний, без вставок типа "См. также" или "Источники".
Пиши нормальным литературным стилем, как в энциклопедии для широкой аудитории.
Не используй разметку, скобочные ссылки и другие формальности Википедии.
Просто цельный, аккуратный, логичный текст на русском языке.
"""


# --- Шаг 1: Загрузка и фильтрация данных из входного файла части ---
print("="*50)
print(f"Шаг 1: Загрузка и фильтрация данных из входного файла '{input_jsonl_path}'...")

items_to_generate = []
total_input_items = 0
skipped_unclear_items = 0
skipped_other_status = 0
error_input_lines = 0

# Определяем имя машины из имени входного файла части
try:
    input_filename = os.path.basename(input_jsonl_path)
    # Ожидаем формат "part_machine_name.jsonl"
    if input_filename.startswith("part_") and input_filename.endswith(".jsonl"):
        machine_name = input_filename[len("part_"):-len(".jsonl")]
        if not machine_name:
             raise ValueError("Имя машины не определено после 'part_'.")
    else:
        raise ValueError(f"Имя входного файла '{input_filename}' не соответствует ожидаемому формату 'part_machine_name.jsonl'.")

    output_filename = output_filename_pattern.format(machine_name=machine_name)
    output_full_path = os.path.join(output_directory, output_filename)
    print(f"Определено имя машины: '{machine_name}'")
    print(f"Выходной файл для этой машины: '{output_full_path}'")

except ValueError as ve:
    print(f"\nКритическая ошибка: Не удалось определить имя машины из входного файла '{input_jsonl_path}'.")
    print(f"Пожалуйста, убедитесь, что имя файла соответствует формату 'part_machine_name.jsonl'. Ошибка: {ve}")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка при обработке имени входного файла: {e}")
    traceback.print_exc()
    sys.exit(1)


try:
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            total_input_items += 1
            try:
                item = json.loads(line)

                # Проверяем наличие необходимых полей для генерации статьи
                required_keys = ['number', 'title', 'description', 'description_status']
                if not all(key in item for key in required_keys):
                     print(f"  Пропущена строка {line_num + 1}: Отсутствуют необходимые поля {required_keys}.")
                     skipped_other_status += 1
                     continue

                # Проверяем статус описания
                status = item.get('description_status')
                if SKIP_UNCLEAR_SEEDS and status != 'ok':
                    if status == 'unclear':
                        skipped_unclear_items += 1
                    else:
                        skipped_other_status += 1 # Ошибки API или парсинга на шаге описаний
                    # print(f"  Пропущена строка {line_num + 1}: Статус описания '{status}'.")
                    continue

                # Если статус 'ok' или SKIP_UNCLEAR_SEEDS=False
                items_to_generate.append(item)


            except json.JSONDecodeError:
                print(f"\n  Пропущена строка {line_num + 1}: Ошибка парсинга JSON.")
                error_input_lines += 1
            except Exception as e:
                print(f"\n  Пропущена строка {line_num + 1}: Непредвиденная ошибка: {e}")
                # traceback.print_exc()
                error_input_lines += 1

    print("Шаг 1 завершен.")
    print(f"Всего записей во входном файле: {total_input_items}")
    print(f"Записей для генерации статьи (статус 'ok'): {len(items_to_generate)}")
    print(f"  Пропущено записей (статус 'unclear'): {skipped_unclear_items}")
    print(f"  Пропущено записей (другой статус ошибки): {skipped_other_status}")
    if error_input_lines > 0:
        print(f"  Строк с ошибками во входном файле: {error_input_lines}")


except FileNotFoundError:
    print(f"\nКритическая ошибка: Входной файл '{input_jsonl_path}' не найден.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка во время Шага 1: {e}")
    traceback.print_exc()
    sys.exit(1)

if len(items_to_generate) == 0:
    print("\nНет записей со статусом 'ok' для генерации статей. Скрипт завершен.")
    sys.exit(0)


# --- Шаг 2: Генерация полных статей с помощью LMStudio API и сохранение результатов ---
print("\n" + "="*50)
print("Шаг 2: Генерация полных статей с помощью LMStudio API и сохранение результатов...")
print(f"API URL: {LMSTUDIO_API_URL}")
print(f"Модель: {MODEL_NAME_IN_LMSTUDIO}")
print(f"Выходной файл: '{output_full_path}'")
print(f"Всего статей для генерации: {len(items_to_generate)}")
print(f"Безопасный лимит токенов для статьи: {MAX_TOKENS_ARTICLE}")


generated_count = 0
api_errors_count = 0
parse_errors_count = 0

# Создаем выходную директорию, если она не существует
try:
    os.makedirs(output_directory, exist_ok=True)
    print(f"Директория '{output_directory}' готова.")
except Exception as e:
    print(f"\nКритическая ошибка: Не удалось создать директорию для выходного файла '{output_directory}': {e}")
    sys.exit(1)

# Проверяем доступность API перед началом
try:
    response = requests.get(f"{LMSTUDIO_API_URL.rsplit('/', 1)[0]}/models") # Пробуем получить список моделей или просто пингануть
    response.raise_for_status()
    print("LMStudio API доступен.")
except requests.exceptions.RequestException as req_err:
    print(f"\nКритическая ошибка: Не удалось подключиться к LMStudio API по адресу {LMSTUDIO_API_URL}.")
    print(f"Пожалуйста, убедитесь, что LMStudio запущен и модель '{MODEL_NAME_IN_LMSTUDIO}' загружена и API сервер активен.")
    print(f"Ошибка: {req_err}")
    sys.exit(1)


try:
    # Открываем выходной файл для записи
    with open(output_full_path, 'w', encoding='utf-8') as outfile:

        for i, item in enumerate(items_to_generate):
            # Исходные данные затравки для сохранения
            original_seed_info = item # Сохраняем всю входную запись как инфо о затравке

            title = item.get('title', '')
            description = item.get('description', '')

            # Формируем полный текст промпта для текущей записи
            full_prompt_article = PROMPT_TEMPLATE_ARTICLE.format(
                title=title,
                description=description
            )

            # Формируем тело запроса к API в формате OpenAI Chat Completions
            api_payload = {
                "model": MODEL_NAME_IN_LMSTUDIO,
                "messages": [
                    {"role": "user", "content": full_prompt_article}
                ],
                # Температура не задается здесь, используется глобальная из LMStudio
                # "max_tokens": MAX_TOKENS_ARTICLE, # Безопасный лимит токенов
                # Другие полезные параметры могут быть добавлены, например, stop sequences
            }

            generated_text = ""
            generation_status = "api_error" # Статус по умолчанию

            try:
                # Отправляем запрос к LMStudio API
                response = requests.post(LMSTUDIO_API_URL, json=api_payload)
                response.raise_for_status() # Вызовет исключение для плохих статусов

                # Парсим ответ
                response_json = response.json()
                if response_json and 'choices' in response_json and len(response_json['choices']) > 0:
                    generated_text = response_json['choices'][0]['message']['content']

                    # Очищаем от тегов <think>...</think>
                    cleaned_text = THINK_TAG_REGEX.sub('', generated_text)
                    cleaned_text = cleaned_text.strip()

                    if cleaned_text: # Проверяем, что сгенерированный текст не пустой
                         generation_status = "ok"
                         generated_count += 1
                    else:
                         generation_status = "empty_response" # Модель сгенерировала пустоту после чистки
                         parse_errors_count += 1 # Считаем как ошибку парсинга/обработки

                else:
                    # Ответ API не содержит ожидаемой структуры choices/message
                    generation_status = "parse_error"
                    parse_errors_count += 1
                    print(f"\n  [{i+1}/{len(items_to_generate)}] Ошибка парсинга ответа API для номера {item.get('number')}: Неожиданная структура ответа.")
                    # print(response_json) # Опционально для отладки

            except requests.exceptions.RequestException as req_err:
                # Ошибки запроса (соединение, таймаут, HTTP ошибки)
                api_errors_count += 1
                generation_status = "api_error"
                print(f"\n  [{i+1}/{len(items_to_generate)}] API Ошибка для номера {item.get('number')}: {req_err}")

            except Exception as e:
                 # Другие ошибки при обработке ответа или парсинге
                 parse_errors_count += 1
                 generation_status = "parse_error"
                 print(f"\n  [{i+1}/{len(items_to_generate)}] Ошибка при обработке ответа для номера {item.get('number')}: {e}")
                 # traceback.print_exc() # Опционально для подробной отладки


            # Сохраняем результат обработки этой записи
            output_item = {
                "original_seed_info": original_seed_info, # Вся исходная информация о затравке
                "generated_text": cleaned_text,         # Сгенерированный текст статьи (или пустая строка)
                "generation_status": generation_status,   # Статус генерации ('ok', 'api_error', 'parse_error', 'empty_response')
                "source": "wiki_generated"                # Указываем источник данных
            }

            json_line = json.dumps(output_item, ensure_ascii=False)
            outfile.write(json_line + '\n')

            # Выводим прогресс
            if (i + 1) % 50 == 0: # Выводим прогресс каждые 50 статей
                 print(f"  Обработано {i + 1}/{len(items_to_generate)}. Сгенерировано: {generated_count}, API ошибки: {api_errors_count}, Ошибки парсинга/пусто: {parse_errors_count}")

            # Добавляем задержку между запросами
            time.sleep(RATE_LIMIT_DELAY)

        # --- Конец цикла по записям ---

    print("\nШаг 2 завершен.")
    print(f"Всего записей обработано для генерации: {len(items_to_generate)}")
    print(f"  Успешно сгенерировано статей: {generated_count}")
    print(f"  Ошибки API запросов: {api_errors_count}")
    print(f"  Ошибки парсинга/пустой ответ: {parse_errors_count}")
    print(f"Результаты сохранены в файл: '{output_full_path}'")


except FileNotFoundError:
    print(f"\nКритическая ошибка: Входной файл '{input_jsonl_path}' не найден.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка во время Шага 2: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nСкрипт генерации полных статей завершил работу.")
print("="*50)