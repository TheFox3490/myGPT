# -*- coding: utf-8 -*-

import json
import os
import traceback
import sys
import numpy as np
import collections
import random
import re # Импортируем re для регулярных выражений
# Импортируем AutoTokenizer для загрузки токенизатора из Hugging Face
from transformers import AutoTokenizer

# --- Конфигурация ---
# Список словарей, описывающих входные источники данных
# 'path': путь к файлу JSONL или папке с файлами JSONL
# 'format': 'generated_original', 'wiki_original', 'wiki_generated'
input_sources = [
    {'path': '../myGPTdistr/generated_articles_jsonl', 'format': 'generated_original'},
    {'path': '../myGPTWiki/selected_wiki_jsonl/selected_wiki_articles.jsonl', 'format': 'wiki_original'},
    {'path': '../myGPTWiki/generated_wiki_articles', 'format': 'wiki_generated'},
]

# Папка для сохранения выходных бинарных файлов train.bin, val.bin и meta.json
output_dir = 'data/custom_corpus'

# Соотношение данных для обучения и валидации (например, 0.9 для 90% train, 10% val)
train_val_split = 0.9

# Имя модели на Hugging Face для загрузки токенизатора
MODEL_NAME = "google/gemma-3-27b-it"

# Статус генерации, который считается "успешным" для сгенерированных статей
GENERATION_SUCCESS_STATUS = "ok"

# Регулярное выражение для удаления тегов <think>...</think>
THINK_TAG_REGEX = re.compile(r'<think>.*?</think>', re.DOTALL)

# --- Конец Конфигурации ---

# --- Шаг 1: Загрузка и настройка токенизатора ---
print("="*50)
print(f"Шаг 1: Загрузка токенизатора для модели '{MODEL_NAME}'...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Токенизатор загружен успешно.")

    if tokenizer.bos_token_id is None:
         print(f"\nКритическая ошибка: Токенизатор '{MODEL_NAME}' не имеет стандартного <bos> токена.")
         sys.exit(1)
    if tokenizer.eos_token_id is None:
         print(f"\nКритическая ошибка: Токенизатор '{MODEL_NAME}' не имеет стандартного <eos> токена.")
         sys.exit(1)

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    print(f"  Размер словаря (vocab size): {vocab_size}")
    print(f"  ID токена начала последовательности (<bos>): {bos_token_id}")
    print(f"  ID токена конца последовательности (<eos>): {eos_token_id}")

except Exception as e:
    print(f"\nКритическая ошибка: Не удалось загрузить или настроить токенизатор для модели '{MODEL_NAME}'.")
    print(f"Убедитесь, что у вас установлен transformers (`pip install transformers`) и huggingface_hub (`pip install huggingface_hub`),")
    print(f"есть доступ к интернету и при необходимости выполнена аутентификация (`huggingface-cli login`).")
    print(f"Ошибка: {e}")
    traceback.print_exc()
    sys.exit(1)

print("="*50)

# --- Шаг 2: Подготовка выходных файлов и директории ---
print("\n" + "="*50)
print(f"Шаг 2: Подготовка выходной директории '{output_dir}' и бинарных файлов...")

try:
    os.makedirs(output_dir, exist_ok=True)
    train_filepath = os.path.join(output_dir, 'train.bin')
    val_filepath = os.path.join(output_dir, 'val.bin')

    # Открываем файлы в бинарном режиме для записи
    # Используем with open для гарантии закрытия файлов
    train_file = open(train_filepath, 'wb')
    val_file = open(val_filepath, 'wb')


    print(f"Директория '{output_dir}' готова.")
    print(f"Файлы '{train_filepath}' и '{val_filepath}' открыты для записи.")

except Exception as e:
    print(f"\nКритическая ошибка: Не удалось подготовить выходную директорию или файлы.")
    print(f"Ошибка: {e}")
    traceback.print_exc()
    sys.exit(1)

print("="*50)


# --- Шаг 3: Чтение, токенизация, разделение и запись статей ---
print("\n" + "="*50)
print("Шаг 3: Чтение, токенизация, разделение и запись статей...")

total_articles_processed = 0 # Всего статей, которые скрипт ПЫТАЛСЯ обработать (попыток чтения строк из файлов)
total_articles_successfully_processed = 0 # Всего статей, из которых успешно извлечен текст И они успешно токенизированы
total_articles_skipped = 0 # <<< ИНИЦИАЛИЗАЦИЯ ЗДЕСЬ
token_counts = collections.defaultdict(int) # Для подсчета токенов текста статьи (без BOS/EOS) по источникам
article_counts = collections.defaultdict(int) # Для подсчета статей по источникам (только успешно обработанные)
output_token_counts = {'train': 0, 'val': 0} # Для подсчета токенов в train/val файлах (включая BOS/EOS)


for source_config in input_sources:
    source_path = source_config['path']
    source_format = source_config['format']
    print(f"\nОбработка источника: '{source_path}' (Формат: {source_format})")

    file_list = []
    if os.path.isdir(source_path):
        file_list = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.jsonl')]
        if not file_list:
             print(f"  Внимание: В папке '{source_path}' не найдено файлов .jsonl. Пропускаем источник.")
             continue
        print(f"  Найдено {len(file_list)} файл(ов) в папке.")
    elif os.path.isfile(source_path) and source_path.endswith('.jsonl'):
        file_list = [source_path]
        print(f"  Обрабатывается один файл.")
    else:
        print(f"  Внимание: Путь '{source_path}' не является папкой или файлом .jsonl. Пропускаем источник.")
        continue


    for file_path in file_list:
        # print(f"  Чтение файла: '{file_path}'") # Слишком много вывода при большом количестве файлов
        articles_in_file_processed = 0
        articles_in_file_successfully_processed = 0
        error_in_file_lines = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    total_articles_processed += 1
                    articles_in_file_processed += 1

                    article_text_raw = None # Исходный текст из JSON
                    article_text_cleaned = None # Очищенный текст для токенизации
                    skip_reason = None

                    try:
                        item = json.loads(line)

                        # Логика извлечения и ПЕРВИЧНОЙ очистки текста в зависимости от формата
                        if source_format == 'generated_original':
                            article_text_raw = item.get('text')
                            # УДАЛЯЕМ <think> блоки из generated_original
                            if isinstance(article_text_raw, str):
                                article_text_cleaned = THINK_TAG_REGEX.sub('', article_text_raw).strip()
                            else:
                                skip_reason = "text field missing or not string"


                        elif source_format == 'wiki_original':
                            article_text_raw = item.get('text')
                            # Для wiki_original просто берем текст и чистим пробелы
                            if isinstance(article_text_raw, str):
                                article_text_cleaned = article_text_raw.strip()
                            else:
                                skip_reason = "text field missing or not string"


                        elif source_format == 'wiki_generated':
                            # Для сгенерированных статей проверяем статус и удаляем <think> блоки (уже должно быть сделано, но повторим для безопасности)
                            status = item.get('generation_status')
                            if status == GENERATION_SUCCESS_STATUS:
                                article_text_raw = item.get('generated_text')
                                if isinstance(article_text_raw, str):
                                     article_text_cleaned = THINK_TAG_REGEX.sub('', article_text_raw).strip()
                                else:
                                    skip_reason = f"generated_text missing or not string (status '{status}')"
                            else:
                                skip_reason = f"status is '{status}' != '{GENERATION_SUCCESS_STATUS}'"

                        else:
                             skip_reason = f"unknown format '{source_format}'"

                        # Дополнительная проверка: текст не должен быть пустым или слишком коротким после очистки
                        if article_text_cleaned and len(article_text_cleaned) < 50: # Минимальная длина текста, например, 50 символов
                             skip_reason = f"text too short after cleaning ({len(article_text_cleaned)} chars)"
                             article_text_cleaned = None # Сбрасываем текст, если слишком короткий
                        elif not article_text_cleaned and skip_reason is None:
                             # Если текст None/пуст после очистки, но skip_reason не установлен (напр. исходно был пуст)
                             skip_reason = "text empty after cleaning"


                    except json.JSONDecodeError:
                        skip_reason = "JSONDecodeError"
                        error_in_file_lines += 1
                    except Exception as e:
                        skip_reason = f"ExtractionError: {e}"
                        error_in_file_lines += 1
                        # traceback.print_exc() # Опционально


                    # Если очищенный текст статьи доступен и не пропущен
                    if article_text_cleaned:
                        articles_in_file_successfully_processed += 1
                        total_articles_successfully_processed += 1

                        try:
                            # Токенизация текста статьи
                            # add_special_tokens=False, чтобы не добавлять стандартные BOS/EOS от encode
                            article_token_ids = tokenizer.encode(article_text_cleaned, add_special_tokens=False)

                            if not article_token_ids:
                                skip_reason = "empty token_ids after encoding"
                                # Если нет токенов, это ошибка обработки
                                # print(f"  Внимание: Статья {total_articles_processed} дала 0 токенов после кодирования.") # Отладочное сообщение
                                total_articles_skipped += 1 # Считаем пропущенной
                                continue # Переходим к следующей статье

                            # Подсчет токенов для статистики по источникам (без BOS/EOS)
                            token_counts[source_format] += len(article_token_ids)
                            article_counts[source_format] += 1 # Считаем только успешно токенизированные статьи

                            # Случайное разделение на train/val
                            if random.random() < train_val_split:
                                target_file = train_file
                                split_type = 'train'
                            else:
                                target_file = val_file
                                split_type = 'val'

                            # Формируем последовательность ID для записи: <bos> + текст + <eos>
                            ids_to_write = [bos_token_id] + article_token_ids + [eos_token_id]

                            # Записываем последовательность ID в бинарный файл
                            # ИСПОЛЬЗУЕМ np.uint32
                            np.array(ids_to_write, dtype=np.uint32).tofile(target_file)

                            # Подсчет токенов в выходных файлах (включая BOS/EOS)
                            output_token_counts[split_type] += len(ids_to_write)

                        except Exception as e:
                            skip_reason = f"Tokenization/WritingError: {e}"
                            error_in_file_lines += 1
                            total_articles_skipped += 1 # Считаем пропущенной
                            # print(f"  Ошибка при токенизации/записи статьи {total_articles_processed}: {e}") # Отладочное сообщение
                            # traceback.print_exc() # Опционально

                    # Если статья была пропущена на этапе извлечения/очистки текста
                    elif skip_reason:
                         total_articles_skipped += 1
                         # Опционально: выводить причину пропуска для каждой статьи, если нужно отлаживать
                         # print(f"  Пропущена статья {total_articles_processed} (из файла {os.path.basename(file_path)}, строка {line_num + 1}): {skip_reason}")


            # Статистика по текущему файлу
            print(f"  Завершено чтение файла '{os.path.basename(file_path)}'. Всего записей в файле: {articles_in_file_processed}, Успешно обработано: {articles_in_file_successfully_processed}, Ошибки чтения/парсинга: {error_in_file_lines}")


        except FileNotFoundError:
            print(f"  Ошибка: Файл не найден '{file_path}'. Пропускаем.")
            # articles_in_file_processed статей из этого файла не были учтены в total_articles_processed
            # Статьи, пропущенные в этом файле, уже должны были быть учтены в total_articles_skipped
            pass

        except Exception as e:
            print(f"  Критическая ошибка при чтении файла '{file_path}': {e}")
            traceback.print_exc()
            # Статьи, пропущенные в этом файле, уже должны были быть учтены в total_articles_skipped
            pass # Продолжаем с другими файлами/источниками


# --- Шаг 4: Завершение и вывод статистики ---
print("\n" + "="*50)
print("Шаг 4: Завершение обработки и вывод статистики...")

# Закрываем бинарные файлы
try:
    train_file.close()
    val_file.close()
    print("Бинарные файлы train.bin и val.bin успешно закрыты.")
except Exception as e:
     print(f"Ошибка при закрытии бинарных файлов: {e}")

# Сохраняем meta.json
meta_filepath = os.path.join(output_dir, 'meta.json')
meta_info = {
    'vocab_size': vocab_size,
    'bos_token_id': bos_token_id,
    'eos_token_id': eos_token_id,
    'source_files': [src['path'] for src in input_sources],
    'train_val_split': train_val_split,
    'tokenizer_model': MODEL_NAME,
    'total_articles_processed_attempts': total_articles_processed,
    'total_articles_successfully_processed': total_articles_successfully_processed,
    'total_articles_skipped': total_articles_skipped, # ЭТА ПЕРЕМЕННАЯ ТЕПЕРЬ ИНИЦИАЛИЗИРОВАНА
    'total_tokens_in_output_including_special': output_token_counts['train'] + output_token_counts['val'],
    'train_token_count': output_token_counts['train'],
    'val_token_count': output_token_counts['val'],
    'average_tokens_per_article_by_source_without_special': {}, # Будет заполнено ниже
    'article_counts_by_source': dict(article_counts), # Сохраняем количество успешно обработанных статей по источникам
    'token_counts_by_source_without_special': dict(token_counts), # Сохраняем количество токенов по источникам (без BOS/EOS)
}

# Расчет средней длины статьи в токенах (без учета <bos>/<eos>) по источникам
print("\nСредняя длина статьи в токенах (без учета <bos>/<eos>) по источникам:")
for source_config in input_sources:
    fmt = source_config['format']
    num_articles = article_counts.get(fmt, 0)
    num_tokens = token_counts.get(fmt, 0)
    if num_articles > 0:
        avg_tokens = num_tokens / num_articles
        meta_info['average_tokens_per_article_by_source_without_special'][fmt] = avg_tokens
        print(f"  '{fmt}': {avg_tokens:.2f} токенов/статья ({num_articles} статей)")
    else:
        meta_info['average_tokens_per_article_by_source_without_special'][fmt] = 0
        print(f"  '{fmt}': Нет успешно обработанных статей из этого источника.")


try:
    with open(meta_filepath, 'w', encoding='utf-8') as meta_f:
        json.dump(meta_info, meta_f, ensure_ascii=False, indent=4)
    print(f"\nФайл метаданных '{meta_filepath}' сохранен.")
except Exception as e:
    print(f"\nОшибка при сохранении файла метаданных: {e}")


print("\nОбщая статистика обработки:")
print(f"Всего статей обработано (попыток чтения): {total_articles_processed}")
print(f"Всего статей успешно обработано и включено в корпус: {total_articles_successfully_processed}")
print(f"Всего статей пропущено (ошибки, статус, слишком короткие): {total_articles_skipped}")


print("\nКоличество токенов в выходных файлах (включая <bos>/<eos>):")
print(f"  train.bin: {output_token_counts['train']} токенов")
print(f"  val.bin:   {output_token_counts['val']} токенов")
total_output_tokens = output_token_counts['train'] + output_token_counts['val']
print(f"  Всего записано токенов: {total_output_tokens}")

print("\nСкрипт prepare.py завершил работу.")
print("Дальнейшие действия:")
print(f"1. Убедитесь, что в '{output_dir}' созданы файлы train.bin, val.bin и meta.json.")
print("2. Используйте информацию из meta.json для настройки конфигурации модели nanoGPT (в файле config.py или аналогичном):")
print(f"   - Размер словаря (vocab_size) должен быть: {meta_info['vocab_size']}")
print("   - ID токенов <bos> и <eos> (если нужны в логике модели/сэмплинга):")
print(f"     bos_token_id = {meta_info['bos_token_id']}")
print(f"     eos_token_id = {meta_info['eos_token_id']}")
print("   - Укажите путь к данным: data_dir = '{output_dir}'")
print("   - Подберите параметры модели (n_layer, n_embd, n_head) так, чтобы общее количество параметров было ~50-60M с учетом нового vocab_size.")
print("3. Настройте скрипт сэмплирования (sample.py), чтобы он использовал токенизатор")
print(f"   '{MODEL_NAME}' и знал ID <bos>/<eos> (из meta.json) для декодирования.")
print("4. Запускайте тренировку nanoGPT.")

print("="*50)