# -*- coding: utf-8 -*-

import json
import os
import traceback
import sys
import time
import requests # Импортируем библиотеку для HTTP запросов
import re       # Импортируем для обработки тегов <think>

# --- Конфигурация ---
# Путь к файлу с подготовленными данными для генерации описаний
# Создан предыдущим скриптом 10_prepare_description_seeds.py
input_jsonl_path = "./wiki_description_seeds_split/part_3080_machine.jsonl"

# Папка и имя файла для сохранения результатов генерации описаний
output_directory = "./wiki_seeds_with_descriptions"
output_filename = "wiki_seeds_with_descriptions.jsonl"
output_full_path = os.path.join(output_directory, output_filename)

# --- Параметры LMStudio API ---
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions" # Уточните в LMStudio
MODEL_NAME_IN_LMSTUDIO = "qwen3-8b" # Уточните точное имя вашей модели в LMStudio

# Параметры генерации описаний
TEMPERATURE = 0.7
MAX_TOKENS_DESCRIPTION = 100 # Максимальное количество токенов для описания (1-3 предложения)

# Маркер, который модель возвращает в случае неясности
UNCLEAR_MARKER = "НЕЯСНО"

# Задержка между запросами к API (в секундах), чтобы не перегружать LMStudio
RATE_LIMIT_DELAY = 0.1 # Начните с 0.1 или 0.5, если возникают ошибки связи


# Пример текста промпта (будет форматироваться данными из файла)
# Изменяем формулировку про категорию и условие "НЕЯСНО"

PROMPT_TEMPLATE = """Ты — ассистент, помогающий составить краткое описание для энциклопедии.
Твоя задача: прочитать заголовок статьи и предоставленное начало оригинального текста, понять основную тему и сгенерировать краткое описание этой статьи в 1-3 предложения. Описание должно быть информативным и передавать суть статьи.

<category>Предполагаемая категория статьи (используй для общего контекста, но НЕ считай обязательным): {predicted_category}</category>

---
<title>{title}</title>

<beginning_of_text>
{beginning_of_text}
</beginning_of_text>
---

Основываясь ТОЛЬКО на содержании, представленном в <title> и <beginning_of_text>, определи тему статьи. Если это содержание недостаточно ясно или кажется бессмысленным для определения темы, ответь ТОЛЬКО словом "{unclear_marker}".

Генерируй ТОЛЬКО описание (1-3 предложения) или слово "{unclear_marker}". Не добавляй никаких других слов, вступлений или заключений.

Описание:"""

# Регулярное выражение для удаления тегов <think>...</think>
THINK_TAG_REGEX = re.compile(r'<think>.*?</think>', re.DOTALL)
# --- Конец Конфигурации ---


# --- Шаг 1: Загрузка подготовленных данных для генерации ---
print("="*50)
print(f"Шаг 1: Загрузка подготовленных данных для генерации описаний из '{input_jsonl_path}'...")

seeds_for_description_gen = []
processed_input_count = 0
error_input_lines = 0

try:
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line)
                # Проверяем наличие необходимых полей
                if all(key in item for key in ['number', 'predicted_category', 'title', 'beginning_of_text']):
                     seeds_for_description_gen.append(item)
                     processed_input_count += 1
                else:
                    print(f"  Пропущена строка {line_num + 1} во входном файле: Отсутствуют необходимые поля.")
                    error_input_lines += 1

            except json.JSONDecodeError:
                print(f"\n  Пропущена строка {line_num + 1} во входном файле: Ошибка парсинга JSON.")
                error_input_lines += 1
            except Exception as e:
                print(f"\n  Пропущена строка {line_num + 1} во входном файле: Непредвиденная ошибка: {e}")
                # traceback.print_exc()
                error_input_lines += 1

    print("Шаг 1 завершен.")
    print(f"Всего записей для генерации описаний загружено: {processed_input_count}")
    if error_input_lines > 0:
        print(f"  Строк с ошибками во входном файле: {error_input_lines}")


except FileNotFoundError:
    print(f"\nКритическая ошибка: Входной файл '{input_jsonl_path}' не найден.")
    print("Пожалуйста, убедитесь, что предыдущий скрипт успешно его создал.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка во время Шага 1: {e}")
    traceback.print_exc()
    sys.exit(1)

if processed_input_count == 0:
    print("\nНет данных для генерации описаний. Скрипт завершен.")
    sys.exit(0)


# --- Шаг 2: Генерация описаний с помощью LMStudio API и сохранение результатов ---
print("\n" + "="*50)
print("Шаг 2: Генерация описаний с помощью LMStudio API и сохранение результатов...")
print(f"API URL: {LMSTUDIO_API_URL}")
print(f"Модель: {MODEL_NAME_IN_LMSTUDIO}")
print(f"Выходной файл результатов: '{output_full_path}'")
print(f"Всего записей для обработки: {len(seeds_for_description_gen)}")


generated_descriptions_count = 0
api_errors_count = 0
parse_errors_count = 0 # Ошибки при парсинге или очистке ответа модели
unclear_count = 0      # Количество записей, помеченных моделью как НЕЯСНО

# Создаем выходную директорию, если она не существует
try:
    os.makedirs(output_directory, exist_ok=True)
    print(f"Директория '{output_directory}' готова.")
except Exception as e:
    print(f"\nКритическая ошибка: Не удалось создать директорию для выходного файла '{output_directory}': {e}")
    sys.exit(1)


try:
    # Открываем выходной файл для записи
    with open(output_full_path, 'w', encoding='utf-8') as outfile:

        for i, item in enumerate(seeds_for_description_gen):
            number = item.get('number') # Сохраняем оригинальный номер
            predicted_category = item.get('predicted_category')
            title = item.get('title')
            beginning_of_text = item.get('beginning_of_text')

            # Формируем полный текст промпта для текущей записи
            full_prompt = PROMPT_TEMPLATE.format(
                predicted_category=predicted_category,
                title=title,
                beginning_of_text=beginning_of_text,
                unclear_marker=UNCLEAR_MARKER # Передаем маркер в промпт
            )

            # Формируем тело запроса к API в формате OpenAI Chat Completions
            api_payload = {
                "model": MODEL_NAME_IN_LMSTUDIO,
                "messages": [
                    # Можно использовать роль system, но часто user/user + assistant chain тоже работает
                    {"role": "user", "content": full_prompt}
                ],
                #"temperature": TEMPERATURE,
                #"max_tokens": MAX_TOKENS_DESCRIPTION,
                # Другие полезные параметры могут быть добавлены здесь, если нужно
            }

            generated_description = ""
            description_status = "api_error" # Статус по умолчанию

            try:
                # Отправляем запрос к LMStudio API
                response = requests.post(LMSTUDIO_API_URL, json=api_payload)
                response.raise_for_status() # Вызовет исключение для плохих статусов (4xx или 5xx)

                # Парсим ответ
                response_json = response.json()
                if response_json and 'choices' in response_json and len(response_json['choices']) > 0:
                    # Извлекаем сгенерированный текст
                    generated_text = response_json['choices'][0]['message']['content']

                    # Очищаем от тегов <think>...</think>
                    cleaned_text = THINK_TAG_REGEX.sub('', generated_text)
                    cleaned_text = cleaned_text.strip() # Удаляем ведущие/завершающие пробелы

                    # Проверяем на маркер НЕЯСНО
                    if cleaned_text.upper() == UNCLEAR_MARKER.upper():
                        generated_description = "" # Описание пустое, если модель неясна
                        description_status = "unclear"
                        unclear_count += 1
                    else:
                        generated_description = cleaned_text
                        description_status = "ok"
                        generated_descriptions_count += 1

                else:
                    # Ответ API не содержит ожидаемой структуры choices/message
                    description_status = "parse_error"
                    parse_errors_count += 1
                    print(f"\n  [{i+1}/{len(seeds_for_description_gen)}] Ошибка парсинга ответа API для номера {number}: Неожиданная структура ответа.")
                    # print(response_json) # Опционально вывести весь ответ для отладки

            except requests.exceptions.RequestException as req_err:
                # Ошибки запроса (соединение, таймаут, HTTP ошибки)
                api_errors_count += 1
                description_status = "api_error"
                print(f"\n  [{i+1}/{len(seeds_for_description_gen)}] API Ошибка для номера {number}: {req_err}")

            except Exception as e:
                 # Другие ошибки при обработке ответа или парсинге
                 parse_errors_count += 1
                 description_status = "parse_error"
                 print(f"\n  [{i+1}/{len(seeds_for_description_gen)}] Ошибка при обработке ответа для номера {number}: {e}")
                 # traceback.print_exc() # Опционально для подробной отладки


            # Сохраняем результат обработки этой записи
            output_item = {
                "number": number,
                "predicted_category": predicted_category,
                "title": title,
                "beginning_of_text": beginning_of_text, # Сохраняем исходное начало текста
                "description": generated_description,   # Сгенерированное описание (или пустая строка)
                "description_status": description_status# Статус генерации ('ok', 'unclear', 'api_error', 'parse_error')
            }

            json_line = json.dumps(output_item, ensure_ascii=False)
            outfile.write(json_line + '\n')

            # Выводим прогресс
            if (i + 1) % 100 == 0: # Выводим прогресс чаще, так как каждый запрос занимает время
                 print(f"  Обработано {i + 1}/{len(seeds_for_description_gen)}. API ошибки: {api_errors_count}, Ошибки парсинга: {parse_errors_count}, Неясные: {unclear_count}")

            # Добавляем задержку между запросами
            time.sleep(RATE_LIMIT_DELAY)

        # --- Конец цикла по записям ---

    print("\nШаг 2 завершен.")
    print(f"Всего записей обработано: {len(seeds_for_description_gen)}")
    print(f"  Успешно сгенерировано описаний: {generated_descriptions_count}")
    print(f"  Модель ответила '{UNCLEAR_MARKER}': {unclear_count}")
    print(f"  Ошибки API запросов: {api_errors_count}")
    print(f"  Ошибки парсинга/обработки ответа: {parse_errors_count}")
    print(f"Результаты сохранены в файл: '{output_full_path}'")


except FileNotFoundError:
    print(f"\nКритическая ошибка: Входной файл '{input_jsonl_path}' не найден.")
    print("Пожалуйста, убедитесь, что предыдущий скрипт успешно его создал.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка во время Шага 2: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nСкрипт генерации описаний завершил работу.")
print("="*50)