# -*- coding: utf-8 -*-

import json
import os
import traceback
import sys
import collections
# import re # Не нужен для простой эвристики

# --- Конфигурация ---
# Путь к файлу с оригинальными статьями Wiki40b, отобранными по длине (1.6ГБ)
# Нужен для извлечения начала текста по номеру
original_wiki_jsonl_path = "./selected_wiki_jsonl/selected_wiki_articles.jsonl"

# Путь к файлу с отобранными статьями и заголовками (~30к статей)
# Создан предыдущим скриптом 3_sample_and_title_wiki.py
selected_titles_jsonl_path = "./wiki_seed_titles/selected_wiki_titles.jsonl"

# Папка и имя файла для сохранения данных, готовых для генерации описаний
output_directory = "./wiki_seeds_for_description"
output_filename = "wiki_seeds_for_description.jsonl"
output_full_path = os.path.join(output_directory, output_filename)

# Параметр для извлечения "начала статьи"
MAX_CHARS = 1000         # Берем первые N символов после заголовка
# --- Конец Конфигурации ---


# --- Шаг 1: Загрузка списка отобранных статей с заголовками ---
print("="*50)
print(f"Шаг 1: Загрузка списка отобранных статей с заголовками из '{selected_titles_jsonl_path}'...")

selected_articles_info = []
processed_titles_count = 0
error_titles_lines = 0

try:
    with open(selected_titles_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line)
                number = item.get('number')
                category = item.get('predicted_category')
                title = item.get('title')

                if number is None or category is None or title is None:
                     print(f"  Пропущена строка {line_num + 1} в файле заголовков: Неполные данные.")
                     error_titles_lines += 1
                     continue

                selected_articles_info.append({'number': number, 'predicted_category': category, 'title': title})
                processed_titles_count += 1

            except json.JSONDecodeError:
                print(f"\n  Пропущена строка {line_num + 1} в файле заголовков: Ошибка парсинга JSON.")
                error_titles_lines += 1
            except Exception as e:
                print(f"\n  Пропущена строка {line_num + 1} в файле заголовков: Непредвиденная ошибка: {e}")
                # traceback.print_exc()
                error_titles_lines += 1

    print("Шаг 1 завершен.")
    print(f"Всего записей заголовков обработано: {processed_titles_count}")
    if error_titles_lines > 0:
        print(f"  Строк с ошибками в файле заголовков: {error_titles_lines}")


except FileNotFoundError:
    print(f"\nКритическая ошибка: Файл отобранных заголовков '{selected_titles_jsonl_path}' не найден.")
    print("Пожалуйста, убедитесь, что предыдущий скрипт отбора и извлечения заголовков успешно его создал.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка во время Шага 1: {e}")
    traceback.print_exc()
    sys.exit(1)

if processed_titles_count == 0:
    print("\nНет отобранных заголовков для обработки. Скрипт завершен.")
    sys.exit(0)


# --- Шаг 2: Загрузка оригинальных текстов по номерам ---
print("\n" + "="*50)
print(f"Шаг 2: Загрузка оригинальных текстов статей из '{original_wiki_jsonl_path}' для быстрого доступа...")
print(f"ВНИМАНИЕ: Этот файл (~1.6ГБ) будет загружен в оперативную память.")

original_texts_by_number = {}
processed_original_count = 0
error_original_lines = 0

try:
    with open(original_wiki_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line)
                number = item.get('number')
                text = item.get('text')

                if number is None or text is None:
                    # print(f"  Пропущена строка {line_num + 1} в оригинальном файле: Неполные данные.")
                    error_original_lines += 1
                    continue

                original_texts_by_number[number] = text
                processed_original_count += 1

            except json.JSONDecodeError:
                print(f"\n  Пропущена строка {line_num + 1} в оригинальном файле: Ошибка парсинга JSON.")
                error_original_lines += 1
            except Exception as e:
                print(f"\n  Пропущена строка {line_num + 1} в оригинальном файле: Непредвиденная ошибка: {e}")
                # traceback.print_exc()
                error_original_lines += 1

    print("Шаг 2 завершен.")
    print(f"Всего записей оригинальных статей обработано: {processed_original_count}")
    if error_original_lines > 0:
        print(f"  Строк с ошибками в оригинальном файле: {error_original_lines}")
    print(f"Тексты загружены в память для {len(original_texts_by_number)} уникальных номеров.")

except FileNotFoundError:
    print(f"\nКритическая ошибка: Оригинальный файл статей '{original_wiki_jsonl_path}' не найден.")
    print("Пожалуйста, убедитесь, что скрипт отбора по длине успешно его создал.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка во время Шага 2: {e}")
    traceback.print_exc()
    sys.exit(1)

if len(original_texts_by_number) == 0:
    print("\nНе удалось загрузить оригинальные тексты. Скрипт завершен.")
    sys.exit(0)


# --- Шаг 3: Извлечение начала текста (простая эвристика) и подготовка к записи ---
print("\n" + "="*50)
print(f"Шаг 3: Извлечение начала текста (первые {MAX_CHARS} символов после заголовка) и подготовка данных для генерации описаний...")

seeds_for_description_gen = []
processed_count = 0
errors_text_extraction = 0

for item in selected_articles_info:
    number = item.get('number')
    category = item.get('predicted_category')
    title = item.get('title')

    if number is None or category is None or title is None:
        errors_text_extraction += 1
        continue

    full_text = original_texts_by_number.get(number)

    beginning_of_text = ""
    if full_text:
        # Находим позицию первого переноса строки
        first_newline_pos = full_text.find('\n')

        if first_newline_pos != -1:
            # Берем текст после первого переноса строки и до MAX_CHARS
            # Используем strip() на всякий случай для удаления пробелов в начале или конце
            beginning_of_text = full_text[first_newline_pos + 1:].strip()[:MAX_CHARS]
        else:
            # Если переноса строки нет (вся статья - одна строка), считаем началом статьи весь текст (хотя это странно для вики)
            # Или можно взять первые MAX_CHARS от всей строки, но так можем обрезать сам заголовок.
            # Давайте все же считать, что если переноса нет, то нет и тела статьи после заголовка в стандартном вики-смысле.
            # Оставим beginning_of_text пустой строкой в этом случае.
            pass # beginning_of_text уже ""


    # Сохраняем данные, готовые для генерации описания
    seeds_for_description_gen.append({
        'number': number,
        'predicted_category': category,
        'title': title,
        'beginning_of_text': beginning_of_text
    })

    processed_count += 1
    if processed_count % 1000 == 0:
         print(f"  Обработано для извлечения начала текста: {processed_count}/{len(selected_articles_info)}")

print("\nШаг 3 завершен.")
print(f"Подготовлено записей для генерации описаний: {len(seeds_for_description_gen)} (Ошибок при обработке: {errors_text_extraction})")


# --- Шаг 4: Сохранение подготовленных данных ---
print("\n" + "="*50)
print("Шаг 4: Сохранение подготовленных данных для генерации описаний...")
print(f"Выходной файл: '{output_full_path}'")

if not seeds_for_description_gen:
    print("Нечего записывать. Список подготовленных данных пуст.")
else:
    try:
        os.makedirs(output_directory, exist_ok=True)
        with open(output_full_path, 'w', encoding='utf-8') as outfile:
            for i, item in enumerate(seeds_for_description_gen):
                json_line = json.dumps(item, ensure_ascii=False)
                outfile.write(json_line + '\n')

                if (i + 1) % 1000 == 0:
                     print(f"  Записано {i + 1}/{len(seeds_for_description_gen)} записей...")

        print(f"\nШаг 4 завершен. Файл '{output_full_path}' успешно создан.")
        print(f"Всего записей сохранено: {len(seeds_for_description_gen)}")

    except Exception as e:
        print(f"\nКритическая ошибка во время Шага 4 (запись файла): {e}")
        traceback.print_exc()
        sys.exit(1)

print("\nСкрипт подготовки данных для генерации описаний завершил работу.")
print("="*50)