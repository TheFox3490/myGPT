# -*- coding: utf-8 -*-

from transformers import pipeline
import torch
import json
import os
import traceback
import sys
import collections


# --- Конфигурация ---
# Путь к входному JSONL файлу с отобранными статьями wiki40b (результат предыдущего скрипта)
input_jsonl_path = "../myGPTWiki/selected_wiki_jsonl/selected_wiki_articles.jsonl"

# Список категорий-меток для классификации
# Этот список был согласован ранее. Измените, если нужно.
candidate_labels = [
    'Наука', 'Технология', 'Медицина', 'История', 'География', 'Общество',
    'Политика', 'Культура и Искусство', 'Философия и Религия', 'Персона',
    'Концепция или Теория', 'Событие', 'Организация', 'Другое'
]

# Название модели Zero-shot классификации
model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Папка и имя файла для сохранения результатов классификации (номер, категория, оценка)
output_directory = "./classified_wiki_jsonl"
output_filename = "classified_wiki_results.jsonl"
output_full_path = os.path.join(output_directory, output_filename)

# Размер батча для классификации (подберите оптимальное значение для вашей GPU)
BATCH_SIZE = 128 # Можно попробовать 64, 128 и т.д.
# --- Конец Конфигурации ---


# --- Шаг 1: Загрузка классификационного пайплайна ---
print("="*50)
print("Шаг 1: Загрузка Zero-shot классификационного пайплайна...")
print(f"Модель: '{model_name}'")

# Определяем устройство для выполнения: GPU (cuda:0) если доступен, иначе CPU (-1)
device = 0 if torch.cuda.is_available() else -1
print(f"Используется устройство: {'GPU' if device == 0 else 'CPU'}")

try:
    # Создаем пайплайн для zero-shot классификации
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        tokenizer=model_name,
        device=device
    )
    print("Пайплайн загружен успешно.")

except Exception as e:
    print(f"\nКритическая ошибка при загрузке классификационного пайплайна: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Шаг 2: Классификация статей, запись результатов и подсчет по категориям ---
print("\n" + "="*50)
print("Шаг 2: Чтение статей, классификация, запись результатов и подсчет по категориям...")
print(f"Входной файл: '{input_jsonl_path}'")
print(f"Выходной файл результатов: '{output_full_path}'")
print(f"Количество категорий: {len(candidate_labels)}")
print(f"Размер батча: {BATCH_SIZE}")

articles_batch = []
classified_count = 0
error_lines = 0
category_counts = collections.Counter()

# Создаем директорию для выходного файла, если она не существует
try:
    os.makedirs(output_directory, exist_ok=True)
    print(f"Директория '{output_directory}' готова.")
except Exception as e:
    print(f"\nКритическая ошибка: Не удалось создать директорию для выходного файла '{output_directory}': {e}")
    sys.exit(1)


try:
    # Открываем входной и выходной файлы
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_full_path, 'w', encoding='utf-8') as outfile:

        print("Файлы открыты. Начинается обработка...")
        # Читаем входной файл построчно
        for line_num, line in enumerate(infile):
            try:
                # Парсим JSON объект из строки
                article_data = json.loads(line)

                # Извлекаем нужные поля
                article_number = article_data.get('number', f'error_{line_num}')
                article_text = article_data.get('text')

                if article_text is None:
                    print(f"  Пропущена строка {line_num + 1}: Отсутствует ключ 'text'.")
                    error_lines += 1
                    continue

                # Добавляем статью в текущий батч
                articles_batch.append({'number': article_number, 'text': article_text})

                # Если батч заполнен
                if len(articles_batch) == BATCH_SIZE:

                    # --- Выполняем классификацию для текущего батча ---
                    try:
                        # Извлекаем только тексты для классификатора
                        batch_texts = [item['text'] for item in articles_batch]
                        # Выполняем классификацию батча
                        batch_results = classifier(batch_texts, candidate_labels, multi_label=False) # batch_results - это список результатов

                        # --- Обрабатываем результаты батча, обновляем счетчик и записываем ---
                        # batch_results - это список словарей, по одному словарю для каждого текста в батче
                        for i, original_item in enumerate(articles_batch):
                             # Получаем словарь результатов для i-й статьи из списка батча
                             article_specific_results = batch_results[i] # ИСПРАВЛЕНО ЗДЕСЬ

                             # Теперь обращаемся к ключам этого словаря
                             predicted_category = article_specific_results['labels'][0] # Предсказанная категория
                             score = article_specific_results['scores'][0]             # Ее вероятность

                             # ОБНОВЛЯЕМ СЧЕТЧИК КАТЕГОРИЙ
                             category_counts[predicted_category] += 1

                             # Формируем словарь для записи
                             result_dict = {
                                 "number": original_item['number'],
                                 "predicted_category": predicted_category,
                                 "score": round(float(score), 4)
                             }

                             # Преобразуем словарь в строку JSON и записываем в файл
                             json_line = json.dumps(result_dict, ensure_ascii=False)
                             outfile.write(json_line + '\n')

                             classified_count += 1

                        # Очищаем батч после обработки
                        articles_batch = []

                        # Выводим прогресс
                        if classified_count % 1000 == 0:
                            print(f"    Обработано и классифицировано {classified_count} статей...")

                    except Exception as classify_batch_error:
                         print(f"\n  Ошибка при классификации батча (начало с строки {line_num + 1 - len(articles_batch)}): {classify_batch_error}")
                         error_lines += len(articles_batch)
                         articles_batch = [] # Очищаем батч

            except json.JSONDecodeError:
                print(f"\n  Пропущена строка {line_num + 1}: Ошибка парсинга JSON.")
                error_lines += 1
            except Exception as e:
                print(f"\n  Пропущена строка {line_num + 1}: Непредвиденная ошибка при чтении или обработке: {e}")
                error_lines += 1

        # --- Обрабатываем оставшийся неполный батч после цикла ---
        if articles_batch:
            print(f"\n  Обработка последнего неполного батча ({len(articles_batch)} статей)...")
            try:
                batch_texts = [item['text'] for item in articles_batch]
                batch_results = classifier(batch_texts, candidate_labels, multi_label=False) # batch_results - это список результатов

                for i, original_item in enumerate(articles_batch):
                    article_specific_results = batch_results[i] # ИСПРАВЛЕНО ЗДЕСЬ

                    predicted_category = article_specific_results['labels'][0]
                    score = article_specific_results['scores'][0]

                    # ОБНОВЛЯЕМ СЧЕТЧИК КАТЕГОРИЙ
                    category_counts[predicted_category] += 1

                    result_dict = {
                        "number": original_item['number'],
                        "predicted_category": predicted_category,
                        "score": round(float(score), 4)
                    }
                    json_line = json.dumps(result_dict, ensure_ascii=False)
                    outfile.write(json_line + '\n')
                    classified_count += 1

                print(f"  Последний батч обработан. Всего классифицировано: {classified_count}")

            except Exception as classify_batch_error:
                 print(f"\n  Ошибка при классификации последнего батча ({len(articles_batch)} статей): {classify_batch_error}")
                 error_lines += len(articles_batch)


    print("\nШаг 2 завершен.")
    print(f"Всего статей обработано и классифицировано: {classified_count}")
    if error_lines > 0:
        print(f"  Строк пропущено из-за ошибок: {error_lines}")
    print(f"Результаты сохранены в файл: '{output_full_path}'")


except FileNotFoundError:
    print(f"\nКритическая ошибка: Входной файл '{input_jsonl_path}' не найден.")
    print("Пожалуйста, убедитесь, что предыдущий скрипт успешно создал этот файл.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка во время Шага 2 (чтение/запись файла): {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Шаг 3: Вывод статистики по категориям ---
print("\n" + "="*50)
print("Шаг 3: Статистика распределения по категориям...")

if classified_count > 0:
    # Сортируем категории по количеству статей (от большего к меньшему)
    sorted_counts = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)

    print(f"\nРаспределение {classified_count} классифицированных статей по категориям:")
    for category, count in sorted_counts:
        percentage = (count / classified_count) * 100 if classified_count > 0 else 0
        print(f"  '{category}': {count} статей ({percentage:.2f}%)")
else:
    print("Нет классифицированных статей для вывода статистики.")

print("\nСкрипт классификации завершил работу.")
print("="*50)