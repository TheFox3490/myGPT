# -*- coding: utf-8 -*-

from transformers import pipeline
import torch
import json
import os
import traceback
import sys
import collections
from datasets import load_dataset

# --- Конфигурация ---
# Путь к входному JSONL файлу с отобранными статьями wiki40b по длине
input_jsonl_path = "../myGPTWiki/selected_wiki_jsonl/selected_wiki_articles.jsonl"

# Путь к файлу, содержащему список категорий для классификации (каждая строка - одна категория)
classes_file_path = "./classes.txt"

# Название модели Zero-shot классификации
model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Папка и имя файла для сохранения результатов классификации (номер, категория, оценка)
output_directory = "./classified_wiki_jsonl"
output_filename = "classified_wiki_results.jsonl"
output_full_path = os.path.join(output_directory, output_filename)

# Размер батча для классификации (используется пайплайном при работе со списком текстов)
# Теперь это явно контролируемый размер батча
BATCH_SIZE = 64 # Начните с 64 или 128, подберите оптимальное для вашей GPU
# --- Конец Конфигурации ---

# --- Шаг 1: Чтение категорий из файла ---
print("="*50)
print(f"Шаг 1: Чтение категорий из файла '{classes_file_path}'...")

candidate_labels = []
try:
    with open(classes_file_path, 'r', encoding='utf-8') as f:
        candidate_labels = [line.strip() for line in f if line.strip()]

    if not candidate_labels:
        print("Критическая ошибка: Файл категорий пуст или не содержит действительных категорий.")
        sys.exit(1)

    print(f"Загружено {len(candidate_labels)} категорий:")
    for label in candidate_labels:
        print(f"- '{label}'")

except FileNotFoundError:
    print(f"\nКритическая ошибка: Файл категорий '{classes_file_path}' не найден.")
    print(f"Пожалуйста, создайте файл и поместите каждую категорию на отдельной строке.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка при чтении файла категорий '{classes_file_path}': {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Шаг 2: Загрузка данных из JSONL в Dataset ---
print("\n" + "="*50)
print(f"Шаг 2: Загрузка данных из '{input_jsonl_path}' в Dataset...")

try:
    dataset_dict = load_dataset('json', data_files=input_jsonl_path)

    if 'train' in dataset_dict:
        dataset = dataset_dict['train']
        print(f"Данные успешно загружены. Найдено {len(dataset)} статей.")

        # --- Принудительное приведение типа колонки 'text' в Dataset (оставляем на всякий случай) ---
        # print("\n--- Приведение типа колонки 'text' в Dataset ---")
        # print(f"Исходная схема Dataset: {dataset.features}")

        # def ensure_text_is_string(example):
        #     example['text'] = str(example.get('text')) if example.get('text') is not None else ""
        #     return example

        # dataset = dataset.map(ensure_text_is_string)
        # print(f"Dataset после приведения типа. Новая схема: {dataset.features}")

        # # Проверяем тип и содержимое нескольких первых примеров после приведения (можно закомментировать после отладки)
        # print("Проверка первых 5 примеров после приведения типа:")
        # for i in range(min(5, len(dataset))):
        #     example = dataset[i]
        #     text_value = example.get('text')
        #     print(f"  Пример {i}: Тип: {type(text_value)}, Значение (начало): '{text_value[:100] if text_value else ''}'")
        #     if not isinstance(text_value, str):
        #         print(f"  Критическая проблема: После приведения типа пример {i} не является строкой! Тип: {type(text_value)}")
        # print("--- Конец проверки типа ---")
        # --- КОНЕЦ Принудительного приведения типа ---


    else:
         print(f"Критическая ошибка: Не найден сплит 'train' в загруженном DatasetDict. Структура: {dataset_dict.keys()}")
         sys.exit(1)

except FileNotFoundError:
    print(f"\nКритическая ошибка: Входной файл '{input_jsonl_path}' не найден.")
    print("Пожалуйста, убедитесь, что предыдущий скрипт успешно создал этот файл.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка при загрузке данных в Dataset: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Шаг 3: Загрузка классификационного пайплайна ---
print("\n" + "="*50)
print("Шаг 3: Загрузка Zero-shot классификационного пайплайна...")
print(f"Модель: '{model_name}'")

device = 0 if torch.cuda.is_available() else -1
print(f"Используется устройство: {'GPU' if device == 0 else 'CPU'}")

try:
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


# --- Шаг 4: Классификация с помощью пайплайна на списке текстов ---
print("\n" + "="*50)
print("Шаг 4: Классификация статей с помощью пайплайна на списке текстов...")
print(f"Запуск классификации {len(dataset)} статей.")
print(f"Размер батча: {BATCH_SIZE}") # Теперь этот BATCH_SIZE используется пайплайном явно

try:
    # --- ИЗВЛЕКАЕМ КОЛОНКУ 'text' КАК СПИСОК ---
    # Это эффективно в datasets, не обязательно загружает весь текст в RAM сразу
    texts_list = dataset['text']
    # Проверяем тип извлеченного списка (для отладки)
    print(f"Тип извлеченной колонки 'text': {type(texts_list)}")
    print(f"Проверка первых 5 типов в извлеченном списке:")
    for i in range(min(5, len(texts_list))):
        print(f"  Тип элемента {i}: {type(texts_list[i])}")
    # --- КОНЕЦ ИЗВЛЕЧЕНИЯ ---


    # --- ВЫЗЫВАЕМ ПАЙПЛАЙН С ИЗВЛЕЧЕННЫМ СПИСКОМ ---
    # Пайплайн сам разобьет этот список на батчи заданного размера
    # multi_label=False означает, что для каждого текста будет один лучший результат
    batch_results_list = classifier(
        texts_list, # Передаем список текстов
        candidate_labels=candidate_labels, # Передаем список категорий
        multi_label=False,
        batch_size=BATCH_SIZE # Передаем размер батча
        # device=device # Устройство уже передано при создании пайплайна
    )
    print("\nКлассификация завершена.")

except Exception as e:
    print(f"\nКритическая ошибка во время классификации: {e}")
    traceback.print_exc()
    sys.exit(1)

# batch_results_list теперь является СПИСКОМ словарей,
# где каждый словарь соответствует одной статье и имеет ключи 'sequence', 'labels', 'scores'.
# Порядок результатов в batch_results_list соответствует порядку текстов в texts_list (и в dataset).

# --- Шаг 5: Обработка результатов и подготовка к записи ---
print("\n" + "="*50)
print("Шаг 5: Обработка результатов и подготовка к записи...")
print(f"Получено {len(batch_results_list)} результатов классификации.")


category_counts = collections.Counter()
results_to_save = []
processed_count = 0

# Проходим по списку результатов классификации
# Используем enumerate, чтобы получить индекс и сопоставить его с оригинальным Dataset
for i, result_dict in enumerate(batch_results_list):
    try:
        # Получаем оригинальный номер из исходного Dataset по тому же индексу
        original_row = dataset[i]
        article_number = original_row.get('number')

        # Результаты классификации для этой статьи находятся в result_dict
        predicted_category = result_dict['labels'][0] # Берем первую (самую вероятную) метку
        score = result_dict['scores'][0]             # Берем ее вероятность

        # ОБНОВЛЯЕМ СЧЕТЧИК КАТЕГОРИЙ
        category_counts[predicted_category] += 1

        # Формируем словарь для записи (только number, категория, оценка)
        result_to_save = {
            "number": article_number,
            "predicted_category": predicted_category,
            "score": round(float(score), 4)
        }
        results_to_save.append(result_to_save) # Добавляем в список для записи

        processed_count += 1
        if processed_count % 1000 == 0:
             print(f"  Обработано для сохранения: {processed_count}")

    except Exception as e:
        # Ошибки здесь менее вероятны, но лучше предусмотреть
        print(f"\n  Ошибка при обработке результата классификации для элемента {i}: {e}. Результат: {result_dict}")
        # traceback.print_exc()
        # Пропускаем этот результат


print(f"\nПодготовка к записи завершена. Готово к записи {len(results_to_save)} результатов.")


# --- Шаг 6: Запись результатов в JSON Lines файл ---
print("\n" + "="*50)
print("Шаг 6: Запись результатов в JSON Lines файл...")
print(f"Выходной файл результатов: '{output_full_path}'")

if not results_to_save:
    print("Нечего записывать. Список результатов пуст.")
else:
    try:
        os.makedirs(output_directory, exist_ok=True)
        with open(output_full_path, 'w', encoding='utf-8') as outfile:
            for i, result_dict in enumerate(results_to_save):
                json_line = json.dumps(result_dict, ensure_ascii=False)
                outfile.write(json_line + '\n')

                if (i + 1) % 1000 == 0:
                     print(f"  Записано {i + 1}/{len(results_to_save)} результатов...")

        print(f"\nШаг 6 завершен. Файл '{output_full_path}' успешно создан.")
        print(f"Всего записано {len(results_to_save)} результатов.")

    except Exception as e:
        print(f"\nКритическая ошибка во время Шага 6 (запись файла): {e}")
        traceback.print_exc()
        sys.exit(1)


# --- Шаг 7: Вывод статистики по категориям ---
print("\n" + "="*50)
print("Шаг 7: Статистика распределения по категориям...")

if processed_count > 0:
    sorted_counts = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)

    print(f"\nРаспределение {processed_count} классифицированных статей по категориям:")
    for category, count in sorted_counts:
        percentage = (count / processed_count) * 100 if processed_count > 0 else 0
        print(f"  '{category}': {count} статей ({percentage:.2f}%)")
else:
    print("Нет классифицированных статей для вывода статистики.")

print("\nСкрипт классификации завершил работу.")
print("="*50)