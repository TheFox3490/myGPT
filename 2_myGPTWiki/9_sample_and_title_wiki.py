# -*- coding: utf-8 -*-

import json
import os
import traceback
import sys
import collections
import random

# --- Конфигурация ---
# Путь к файлу с оригинальными статьями Wiki40b, отобранными по длине (1.6ГБ)
original_wiki_jsonl_path = "./selected_wiki_jsonl/selected_wiki_articles.jsonl"

# Путь к файлу с результатами классификации (номер, категория, оценка)
classified_results_jsonl_path = "../mDeBERTa_classifier/classified_wiki_jsonl/classified_wiki_results.jsonl"

# Папка и имя файла для сохранения списка отобранных статей с заголовками
output_directory = "./wiki_seed_titles"
output_filename = "selected_wiki_titles.jsonl"
output_full_path = os.path.join(output_directory, output_filename)

# --- ПЛАН ВЫБОРКИ: Сколько статей взять из каждой категории ---
# Заполните этот словарь! Ключи - названия категорий из вашего classes.txt.
# Значения - желаемое количество статей для выборки ИЗ ЭТОЙ КОНКРЕТНОЙ КАТЕГОРИИ.
# Общая сумма значений должна составлять 30000.

# Примерное распределение на 30000 статей с учетом приоритетов и статистики классификации:
# Высокий приоритет (Наука, Технология, Мед., Концепция): ~13.8k -> возьмем большую часть
# Средний приоритет (История, География, и т.д.): ~83k -> возьмем умеренно
# Низкий приоритет (Персона, Организация, Другое): ~156k -> возьмем очень мало
category_sampling_counts = {
    'Наука': 4000,              # Из 2558 доступных -> возьмем min(4000, 2558) = 2558
    'Технология': 3000,         # Из 2320 доступных -> возьмем min(3000, 2320) = 2320
    'Медицина': 2000,           # Из 3146 доступных -> возьмем min(2000, 3146) = 2000
    'История': 5000,            # Из 42685 доступных
    'География': 3000,          # Из 14520 доступных
    'Общество': 2000,           # Из 6545 доступных -> возьмем min(2000, 6545) = 2000
    'Политика': 1500,           # Из 6153 доступных
    'Культура и Искусство': 1500,# Из 7334 доступных
    'Философия и Религия': 1000,# Из 5089 доступных
    'Спорт': 500,               # Из 1774 доступных
    'Событие': 2000,            # Из 12293 доступных
    'Организация': 500,         # Из 106562 доступных -> Возьмем мало из этой большой категории
    'Персона': 500,             # Из 16700 доступных -> Возьмем мало биографий
    'Концепция или Теория': 4000, # Из 5753 доступных -> возьмем min(4000, 5753) = 4000
    'Другое': 0                 # Из 33398 доступных -> Возьмем 0 из этой категории
}
# Примерная сумма в этом словаре: 2558+2320+2000+5000+3000+2000+1500+1500+1000+500+2000+500+500+4000+0 = 26878
# Чтобы получить ровно 30000, нужно скорректировать числа, например, увеличить некоторые из Среднего приоритета.
# Например, увеличить Историю до 7000, Географию до 4000, Общество до 3000, Политика до 2000 и Культуру до 2000.
# Новые числа: 4000+3000+2000+7000+4000+3000+2000+2000+1000+500+2000+500+500+4000+0 = 31500 (нужно подбирать точнее)

# ДАВАЙТЕ Я ПРЕДЛОЖУ КОНКРЕТНЫЕ ЧИСЛА, СУММИРУЮЩИЕСЯ ПРИМЕРНО в 30000, С УЧЕТОМ ДОСТУПНОСТИ:
# Доступно: Организация(106562), История(42685), Другое(33398), Персона(16700), География(14520), Событие(12293),
# Культура(7334), Общество(6545), Политика(6153), Концепция(5753), Философия(5089), Медицина(3146),
# Наука(2558), Технология(2320), Спорт(1774)
# Сумма доступных: 266830

# Конкретный план выборки на ~30000 статей:
category_sampling_counts = {
    'Наука': 2558, # Все доступные
    'Технология': 2320, # Все доступные
    'Медицина': 3146, # Все доступные
    'Концепция или Теория': 5753, # Все доступные
    'История': 6000, # Из 42685 доступных
    'География': 4000, # Из 14520 доступных
    'Общество': 3000, # Из 6545 доступных
    'Политика': 2000, # Из 6153 доступных
    'Культура и Искусство': 2000, # Из 7334 доступных
    'Философия и Религия': 1500, # Из 5089 доступных
    'Спорт': 1000, # Из 1774 доступных
    'Событие': 2000, # Из 12293 доступных
    'Организация': 200, # Из 106562 доступных (очень мало)
    'Персона': 200, # Из 16700 доступных (мало биографий)
    'Другое': 0     # Из 33398 доступных (ноль)
}
# Сумма желаемых = 2558 + 2320 + 3146 + 5753 + 6000 + 4000 + 3000 + 2000 + 2000 + 1500 + 1000 + 2000 + 200 + 200 + 0 = 35677
# Получилось чуть больше 30к. Давайте уменьшим пропорционально из категорий Среднего приоритета, чтобы сумма была ближе к 30000.
# Примерно нужно отбросить 5677 статей из категорий Среднего приоритета (~83к).
# Это примерно (5677 / 83000) ~ 6.8%. Уменьшим желаемое количество для Среднего приоритета на 6.8%

category_sampling_counts = {
    'Наука': 2558,
    'Технология': 2320,
    'Медицина': 3146,
    'Концепция или Теория': 5753,
    # Уменьшаем Средний приоритет на ~17%:
    'История': int(6000 * (1 - 0.17)), # ~4980
    'География': int(4000 * (1 - 0.17)), # ~3320
    'Общество': int(3000 * (1 - 0.17)), # ~2490
    'Политика': int(2000 * (1 - 0.17)), # ~1660
    'Культура и Искусство': int(2000 * (1 - 0.17)), # ~1660
    'Философия и Религия': int(1500 * (1 - 0.17)), # ~1245
    'Спорт': int(1000 * (1 - 0.17)), # ~830
    'Событие': int(2000 * (1 - 0.17)), # ~1660
    'Организация': 200,
    'Персона': 200,
    'Другое': 0
}
# Сумма желаемых теперь: 2558+2320+3146+5753 + 4980+3320+2490+1660+1660+1245+830+1660 + 200+200+0 = 31522. Все равно чуть больше.

# Окончательный, подогнанный под 30000 план выборки (пример):
category_sampling_counts = {
    'Наука': 2500,
    'Технология': 2300,
    'Медицина': 3100,
    'Концепция или Теория': 5700,
    'История': 5000,
    'География': 3000,
    'Общество': 2000,
    'Политика': 1500,
    'Культура и Искусство': 1500,
    'Философия и Религия': 1000,
    'Спорт': 800,
    'Событие': 1500,
    'Организация': 100, # Очень мало
    'Персона': 100,     # Мало
    'Другое': 0         # Ноль
}
# Проверяем сумму: 2500+2300+3100+5700+5000+3000+2000+1500+1500+1000+800+1500+100+100+0 = 30100
# Это достаточно близко к 30000. Фактическое количество может быть чуть меньше, если в какой-то категории не хватит статей.
# Используем этот словарь category_sampling_counts в скрипте.
# --- Конец ПЛАНА ВЫБОРКИ ---


# --- Шаг 1: Загрузка результатов классификации и группировка по категориям ---
print("="*50)
print(f"Шаг 1: Загрузка результатов классификации из '{classified_results_jsonl_path}' и группировка по категориям...")

articles_by_category = collections.defaultdict(list)
available_counts = collections.Counter()
processed_classified_count = 0
error_classified_lines = 0

try:
    with open(classified_results_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line)
                category = item.get('predicted_category')
                number = item.get('number')
                score = item.get('score')

                if category is None or number is None or score is None:
                    print(f"  Пропущена строка {line_num + 1} в файле классификации: Неполные данные.")
                    error_classified_lines += 1
                    continue

                articles_by_category[category].append({'number': number, 'score': score})
                available_counts[category] += 1
                processed_classified_count += 1

            except json.JSONDecodeError:
                print(f"\n  Пропущена строка {line_num + 1} в файле классификации: Ошибка парсинга JSON.")
                error_classified_lines += 1
            except Exception as e:
                print(f"\n  Пропущена строка {line_num + 1} в файле классификации: Непредвиденная ошибка: {e}")
                # traceback.print_exc()
                error_classified_lines += 1

    print("Шаг 1 завершен.")
    print(f"Всего записей классификации обработано: {processed_classified_count}")
    if error_classified_lines > 0:
        print(f"  Строк с ошибками в файле классификации: {error_classified_lines}")
    print("Статей доступно по категориям:")
    # Выводим доступное количество для проверки
    for category, count in available_counts.most_common():
         print(f"  '{category}': {count} доступно.")


except FileNotFoundError:
    print(f"\nКритическая ошибка: Файл результатов классификации '{classified_results_jsonl_path}' не найден.")
    print("Пожалуйста, убедитесь, что скрипт классификации успешно его создал.")
    sys.exit(1)
except Exception as e:
    print(f"\nКритическая ошибка во время Шага 1: {e}")
    traceback.print_exc()
    sys.exit(1)

if processed_classified_count == 0:
    print("\nНет данных классификации для обработки. Скрипт завершен.")
    sys.exit(0)

# --- Шаг 2: Загрузка оригинальных текстов по номерам ---
print("\n" + "="*50)
print(f"Шаг 2: Загрузка оригинальных текстов статей из '{original_wiki_jsonl_path}'...")
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
                    print(f"  Пропущена строка {line_num + 1} в оригинальном файле: Неполные данные.")
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

# --- Шаг 3: Выполнение выборки по категориям ---
print("\n" + "="*50)
print("Шаг 3: Выполнение выборки статей по категориям...")
print(f"План выборки (желаемое количество из каждой категории): {category_sampling_counts}")

selected_articles_info = [] # Список для хранения отобранных {number: ..., predicted_category: ...}
actual_sampled_counts = collections.Counter()
total_desired = sum(category_sampling_counts.values())

for category, desired_count in category_sampling_counts.items():
    if desired_count <= 0:
        print(f"  Пропускаем категорию '{category}': Желаемое количество <= 0.")
        continue

    available_items = articles_by_category.get(category, [])
    available_count = len(available_items)

    count_to_sample = min(desired_count, available_count)

    if count_to_sample > 0:
        # Случайная выборка из доступных статей этой категории
        sampled_items = random.sample(available_items, count_to_sample)

        # Добавляем информацию об отобранных статьях (номер, предсказанная категория)
        for item in sampled_items:
             selected_articles_info.append({'number': item['number'], 'predicted_category': category})
             actual_sampled_counts[category] += 1

        print(f"  Категория '{category}': Запрошено {desired_count}, доступно {available_count}, отобрано {count_to_sample}")
    else:
        print(f"  Категория '{category}': Запрошено {desired_count}, доступно {available_count}. Не отобрано статей.")

# Перемешиваем отобранный список
random.shuffle(selected_articles_info)

total_actually_sampled = len(selected_articles_info)
print("\nШаг 3 завершен.")
print(f"Общее количество статей отобрано для извлечения заголовков: {total_actually_sampled} (Запрошено всего: {total_desired})")
if total_actually_sampled < total_desired:
    print(f"ВНИМАНИЕ: Отобрано меньше, чем запрошено, возможно не хватило статей в некоторых категориях.")


# --- Шаг 4: Извлечение заголовков для отобранных статей ---
print("\n" + "="*50)
print("Шаг 4: Извлечение заголовков для отобранных статей...")

final_output_seeds = []
processed_sampled_count = 0
errors_title_extraction = 0

for item in selected_articles_info:
    number = item.get('number')
    category = item.get('predicted_category')

    if number is None or category is None:
        errors_title_extraction += 1
        continue # Пропускаем, если нет номера или категории

    article_text = original_texts_by_number.get(number)

    title = "" # Заголовок по умолчанию - пустая строка
    if article_text:
        # Извлекаем первую строку как заголовок
        lines = article_text.split('\n', 1) # Делим строку максимум на 2 части по первому переносу
        title = lines[0].strip() # Берем первую часть и удаляем пробелы

    # Сохраняем информацию
    final_output_seeds.append({
        'number': number,
        'predicted_category': category,
        'title': title
    })

    processed_sampled_count += 1
    if processed_sampled_count % 1000 == 0:
         print(f"  Обработано для извлечения заголовков: {processed_sampled_count}/{total_actually_sampled}")

print("\nШаг 4 завершен.")
print(f"Всего заголовков извлечено: {len(final_output_seeds)} (Ошибок при извлечении/обработке: {errors_title_extraction})")


# --- Шаг 5: Сохранение отобранных статей с заголовками ---
print("\n" + "="*50)
print("Шаг 5: Сохранение отобранных статей с заголовками...")
print(f"Выходной файл: '{output_full_path}'")

if not final_output_seeds:
    print("Нечего записывать. Список заголовков пуст.")
else:
    try:
        os.makedirs(output_directory, exist_ok=True)
        with open(output_full_path, 'w', encoding='utf-8') as outfile:
            for i, seed_item in enumerate(final_output_seeds):
                json_line = json.dumps(seed_item, ensure_ascii=False)
                outfile.write(json_line + '\n')

                if (i + 1) % 1000 == 0:
                     print(f"  Записано {i + 1}/{len(final_output_seeds)} записей...")

        print(f"\nШаг 5 завершен. Файл '{output_full_path}' успешно создан.")
        print(f"Всего записей сохранено: {len(final_output_seeds)}")

    except Exception as e:
        print(f"\nКритическая ошибка во время Шага 5 (запись файла): {e}")
        traceback.print_exc()
        sys.exit(1)


# --- Шаг 6: Вывод статистики по фактически отобранным статьям ---
print("\n" + "="*50)
print("Шаг 6: Статистика по фактически отобранным статьям (после выборки по категориям)...")

if total_actually_sampled > 0:
    # Используем actual_sampled_counts
    sorted_counts = sorted(actual_sampled_counts.items(), key=lambda item: item[1], reverse=True)

    print(f"\nРаспределение {total_actually_sampled} фактически отобранных статей по категориям:")
    for category, count in sorted_counts:
        percentage = (count / total_actually_sampled) * 100 if total_actually_sampled > 0 else 0
        print(f"  '{category}': {count} статей ({percentage:.2f}%)")
else:
    print("Нет отобранных статей для статистики.")

print("\nСкрипт выборки и извлечения заголовков завершил работу.")
print("="*50)