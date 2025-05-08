# -*- coding: utf-8 -*-

import json
import os
import traceback
import sys
import collections
import random # Не нужен для разделения, но оставим для консистентности с предыдущими скриптами

# --- Конфигурация ---
# Путь к файлу с подготовленными данными для генерации описаний
# Создан скриптом 4_prepare_description_seeds.py
input_jsonl_path = "./wiki_seeds_for_description/wiki_seeds_for_description.jsonl"

# Папка для сохранения разделенных файлов
output_directory = "./wiki_description_seeds_split"

# --- Веса ваших машин ---
# Укажите имена ваших машин (или идентификаторы) и их относительные веса GPU
# Веса примерные, можете скорректировать, если есть более точные данные о производительности на данной задаче.
machine_weights = {
    '3080ti_1': 1.2,
    '3080ti_2': 1.2,
    '3080_machine': 1.0,
    '3070_laptop': 0.75,
}
# --- Конец Конфигурации ---


# --- Шаг 1: Загрузка данных из входного файла ---
print("="*50)
print(f"Шаг 1: Загрузка данных из '{input_jsonl_path}' для разделения...")

loaded_data = []
processed_input_count = 0
error_input_lines = 0

try:
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line)
                # Проверяем наличие хотя бы номера или какого-то ключа, чтобы убедиться, что это валидный объект
                if isinstance(item, dict) and len(item) > 0:
                     loaded_data.append(item)
                     processed_input_count += 1
                else:
                    print(f"  Пропущена строка {line_num + 1}: Некорректный JSON объект или пустая строка.")
                    error_input_lines += 1

            except json.JSONDecodeError:
                print(f"\n  Пропущена строка {line_num + 1}: Ошибка парсинга JSON.")
                error_input_lines += 1
            except Exception as e:
                print(f"\n  Пропущена строка {line_num + 1}: Непредвиденная ошибка: {e}")
                # traceback.print_exc()
                error_input_lines += 1

    print("Шаг 1 завершен.")
    print(f"Всего записей загружено для разделения: {processed_input_count}")
    if error_input_lines > 0:
        print(f"  Строк с ошибками во входном файле: {error_input_lines}")


except FileNotFoundError:
    print(f"\nКритическая ошибка: Входной файл '{input_jsonl_path}' не найден.")
    print("Пожалуйста, убедитесь, что скрипт 4_prepare_description_seeds.py успешно его создал.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка во время Шага 1: {e}")
    traceback.print_exc()
    sys.exit(1)

if processed_input_count == 0:
    print("\nНет данных для разделения. Скрипт завершен.")
    sys.exit(0)

# --- Шаг 2: Расчет количества записей для каждой машины ---
print("\n" + "="*50)
print("Шаг 2: Расчет количества записей для каждой машины...")

total_items = len(loaded_data)
total_weight = sum(machine_weights.values())

if total_weight == 0:
    print("\nКритическая ошибка: Сумма весов машин равна 0. Проверьте конфигурацию machine_weights.")
    sys.exit(1)

items_per_machine = {}
for machine_name, weight in machine_weights.items():
    # Рассчитываем приблизительное количество записей
    items_per_machine[machine_name] = int(weight / total_weight * total_items)

# Корректировка для точного соответствия общему количеству из-за округления
current_sum = sum(items_per_machine.values())
difference = total_items - current_sum

# Добавляем (или отнимаем) разницу от машины с наибольшим весом
if difference != 0:
    machine_with_max_weight = max(machine_weights, key=machine_weights.get)
    items_per_machine[machine_with_max_weight] += difference

print(f"Общее количество записей: {total_items}")
print(f"Суммарный вес GPU: {total_weight}")
print("Планируемое количество записей для каждой машины:")
for machine_name, count in items_per_machine.items():
    print(f"  '{machine_name}': {count} записей")

# Проверяем, что сумма после корректировки точна
if sum(items_per_machine.values()) != total_items:
     print("\nКритическая ошибка: Ошибка при корректировке количества записей. Сумма не совпадает с общим количеством.")
     sys.exit(1)

print("Шаг 2 завершен. Расчеты выполнены успешно.")


# --- Шаг 3: Разделение данных и сохранение в файлы ---
print("\n" + "="*50)
print("Шаг 3: Разделение данных и сохранение в файлы...")
print(f"Выходная папка: '{output_directory}'")

# Создаем выходную папку, если она не существует
try:
    os.makedirs(output_directory, exist_ok=True)
    print(f"Директория '{output_directory}' готова.")
except Exception as e:
    print(f"\nКритическая ошибка: Не удалось создать выходную папку '{output_directory}': {e}")
    sys.exit(1)

current_index = 0
saved_files_count = 0

for machine_name, count in items_per_machine.items():
    if count <= 0:
        print(f"  Пропускаем создание файла для '{machine_name}': Количество записей <= 0.")
        continue

    # Определяем срез данных для текущей машины
    end_index = current_index + count
    split_data = loaded_data[current_index:end_index]

    # Определяем имя выходного файла
    output_filename = f"part_{machine_name}.jsonl"
    output_full_path = os.path.join(output_directory, output_filename)

    try:
        # Сохраняем данные в файл
        with open(output_full_path, 'w', encoding='utf-8') as outfile:
            for i, item in enumerate(split_data):
                json_line = json.dumps(item, ensure_ascii=False)
                outfile.write(json_line + '\n')

        print(f"  Сохранен файл '{output_full_path}' с {len(split_data)} записями.")
        saved_files_count += 1

    except Exception as e:
        print(f"\n  Критическая ошибка: Не удалось сохранить файл '{output_full_path}': {e}")
        # Продолжаем, чтобы попытаться сохранить остальные файлы, но сообщаем об ошибке
        pass

    # Обновляем текущий индекс для следующего среза
    current_index = end_index

print("\nШаг 3 завершен.")
print(f"Всего файлов разделения сохранено: {saved_files_count} из {len(machine_weights)}")


# --- Общий итог ---
print("\n" + "="*50)
print("Скрипт разделения файла завершил работу.")
print("Дальнейшие действия:")
print(f"1. Скопируйте файлы из папки '{output_directory}' на соответствующие машины.")
print(f"   - На Машину 1 (3080 Ti) скопируйте файл 'part_3080ti_1.jsonl'.")
print(f"   - На Машину 2 (3080 Ti) скопируйте файл 'part_3080ti_2.jsonl'.")
print(f"   - На Машину 3 (3080) скопируйте файл 'part_3080_machine.jsonl'.")
print(f"   - На Ноутбук (3070) скопируйте файл 'part_3070_laptop.jsonl'.")
print(f"2. На каждой машине отредактируйте скрипт генерации описаний (12_generate_descriptions.py):")
print(f"   Измените переменную `input_jsonl_path` так, чтобы она указывала на скопированный файл части.")
print(f"   Например, на Машине 1 измените:")
print(f"   `input_jsonl_path = \"{output_directory}/part_3080ti_1.jsonl\"`")
print(f"   Аналогично для других машин.")
print("3. Запустите скрипт 5_generate_descriptions.py на каждой машине параллельно.")
print("="*50)