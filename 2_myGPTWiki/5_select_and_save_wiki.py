from datasets import load_from_disk
import os
import traceback
import ast
import re
import statistics
import json
import sys # Для sys.exit()

# --- Конфигурация ---
# Базовая папка, где сохранены разделы датасета wiki40b
base_wiki_data_directory = "./google_wiki40b_ru"
# Порядок разделов для загрузки и обработки
splits_order = ['train', 'validation', 'test']

# Критерии фильтрации по длине очищенного текста (в символах)
MIN_CLEANED_TEXT_LEN = 2000
MAX_CLEANED_TEXT_LEN = 7000

# Папка и имя файла для сохранения отобранных статей в формате JSONL
output_directory = "./selected_wiki_jsonl"
output_filename = "selected_wiki_articles.jsonl"
# --- Конец конфигурации ---


# --- Функция для очистки текста (та же, что и раньше) ---
def clean_wiki_text(text_with_markers: str) -> str:
    """
    Очищает текст статьи от специфических маркеров wiki40b
    и преобразует его в формат с абзацами.
    """
    cleaned_text = text_with_markers
    cleaned_text = cleaned_text.replace('_NEWLINE_', '\n')
    cleaned_text = cleaned_text.replace('_START_PARAGRAPH_', '\n\n')
    cleaned_text = cleaned_text.replace('_START_SECTION_', '\n\n')
    cleaned_text = cleaned_text.replace('_START_ARTICLE_', '')
    # Удаляем любые оставшиеся одиночные маркеры на всякий случай
    cleaned_text = re.sub(r'_[A-Z_]+_', '', cleaned_text)
    cleaned_text = cleaned_text.strip()
    # Заменяем множественные переносы строк (3 и более) на двойные
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text
# --- Конец функции clean_wiki_text ---


# --- Вспомогательная функция для получения и очистки текста из примера ---
def get_decoded_cleaned_text(example) -> str:
    """
    Извлекает поле 'text' из примера, декодирует его, если нужно, и очищает от маркеров.

    Args:
        example: Один пример (строка) из объекта Dataset.

    Returns:
        str: Очищенный текст статьи, или пустая строка в случае ошибки.
    """
    article_content = example.get('text')
    article_text_decoded = "" # Инициализируем пустой строкой на случай ошибок

    if isinstance(article_content, str):
        try:
            evaluated_content = ast.literal_eval(article_content)
            if isinstance(evaluated_content, bytes):
                try:
                    article_text_decoded = evaluated_content.decode('utf-8')
                except Exception:
                    # print(f"\nПредупреждение (декодирование байт): Не удалось декодировать байты в UTF-8. Содержимое: {evaluated_content[:100]}...")
                    pass # Предполагаем, что ошибки печати не нужны в этом скрипте
            # else:
                # print(f"\nПредупреждение (оценка строки): Оценка строки дала неожиданный тип ({type(evaluated_content)}) вместо байт. Содержимое: {article_content[:100]}...")
        except (ValueError, SyntaxError, TypeError):
            # print(f"\nПредупреждение (оценка строки): Строка не является корректным литералом Python. Содержимое: {article_content[:100]}...")
            pass
        except Exception: # as e:
            # print(f"\nПредупреждение (оценка строки): Непредвиденная ошибка при оценке строки: {e}. Содержимое: {article_content[:100]}...")
            pass

    elif isinstance(article_content, bytes):
        try:
            article_text_decoded = article_content.decode('utf-8')
        except Exception:
            # print(f"\nПредупреждение (декодирование байт напрямую): Не удалось декодировать байты в UTF-8. Содержимое: {article_content[:100]}...")
            pass

    # else:
        # print(f"\nПредупреждение (неожиданный тип): Поле 'text' имеет неожиданный тип: {type(article_content)}. Содержимое: {article_content}")

    if isinstance(article_text_decoded, str):
        cleaned_text = clean_wiki_text(article_text_decoded)
        return cleaned_text
    else:
        return "" # Возвращаем пустую строку, если результат не строка

# --- Конец вспомогательной функции ---


# --- Шаг 1: Загрузка и отбор статей ---
print("="*50)
print("Шаг 1: Загрузка и отбор статей из wiki40b...")
print(f"Базовая директория wiki40b: {base_wiki_data_directory}")
print(f"Критерии длины (символов): от {MIN_CLEANED_TEXT_LEN} до {MAX_CLEANED_TEXT_LEN}")

loaded_splits = {}
selected_articles_cleaned_texts = []
total_articles_processed = 0
articles_passed_filter = 0

try:
    for split_name in splits_order:
        split_path = os.path.join(base_wiki_data_directory, split_name)
        if os.path.exists(split_path):
            print(f"  Обработка раздела '{split_name}'...")
            try:
                dataset_split = load_from_disk(split_path)
                loaded_splits[split_name] = dataset_split # Сохраняем загруженный раздел
                num_examples_in_split = len(dataset_split)

                # Итерируемся по примерам в текущем разделе для отбора
                for i, example in enumerate(dataset_split):
                    total_articles_processed += 1
                    # Получаем и очищаем текст
                    cleaned_text = get_decoded_cleaned_text(example)
                    length = len(cleaned_text)

                    # Проверяем критерии фильтрации
                    if MIN_CLEANED_TEXT_LEN <= length <= MAX_CLEANED_TEXT_LEN:
                         selected_articles_cleaned_texts.append(cleaned_text)
                         articles_passed_filter += 1

                    # Выводим прогресс каждые 10000 статей из общего числа
                    if total_articles_processed % 10000 == 0:
                         print(f"    Обработано {total_articles_processed} статей из всех разделов...")

                print(f"  Обработка раздела '{split_name}' завершена. Отобрано {articles_passed_filter} статей.")

            except Exception as e:
                print(f"\nОшибка при обработке раздела '{split_name}' из '{split_path}': {e}")
                # traceback.print_exc()
        # else:
            # print(f"Папка раздела '{split_path}' не найдена. Пропускаем.") # Сообщение уже есть при загрузке

    print("\nШаг 1 завершен.")
    print(f"Всего статей wiki40b обработано: {total_articles_processed}")
    print(f"Всего статей отобрано по критериям ({MIN_CLEANED_TEXT_LEN}-{MAX_CLEANED_TEXT_LEN} символов): {articles_passed_filter}")

except Exception as e:
    print(f"\nКритическая ошибка во время Шага 1 (загрузка/отбор): {e}")
    traceback.print_exc()
    sys.exit(1) # Завершаем выполнение с ошибкой

if articles_passed_filter == 0:
    print("\nНе найдено статей, соответствующих критериям отбора. Невозможно продолжить.")
    sys.exit(0) # Завершаем выполнение без ошибки


# --- Шаг 2: Подсчет статистики для отобранного подмножества ---
print("\n" + "="*50)
print("Шаг 2: Подсчет статистики для отобранного подмножества wiki40b...")

selected_lengths = [len(text) for text in selected_articles_cleaned_texts]

if not selected_lengths: # Дополнительная проверка, хотя articles_passed_filter уже проверен
    print("Не найдено статей для расчета статистики.")
else:
    selected_lengths.sort()

    overall_mean_length = statistics.mean(selected_lengths)
    overall_median_length = statistics.median(selected_lengths)

    print(f"\nСтатистика отобранных статей wiki40b (в символах):")
    print(f"  Количество отобранных статей: {len(selected_lengths)}")
    print(f"  Средняя длина: {overall_mean_length:.2f}")
    print(f"  Медианная длина: {overall_median_length}")
    print(f"  Самая короткая отобранная статья: {selected_lengths[0]} символов")
    print(f"  Самая длинная отобранная статья: {selected_lengths[-1]} символов")

    percentile = 0.10
    k = max(1, int(percentile * len(selected_lengths)))

    if len(selected_lengths) >= 10:
        shortest_10_percent_lengths = selected_lengths[:k]
        mean_shortest_10 = statistics.mean(shortest_10_percent_lengths)
        print(f"\nСтатистика для {k} ({percentile:.0%}) самых коротких отобранных статей:")
        print(f"  Средняя длина: {mean_shortest_10:.2f}")

        longest_10_percent_lengths = selected_lengths[-k:]
        mean_longest_10 = statistics.mean(longest_10_percent_lengths)
        print(f"\nСтатистика для {k} ({percentile:.0%}) самых длинных отобранных статей:")
        print(f"  Средняя длина: {mean_longest_10:.2f}")
    else:
        print(f"\nНедостаточно отобранных статей ({len(selected_lengths)}) для расчета статистики по 10%.")

print("\nШаг 2 завершен.")


# --- Шаг 3: Пауза и запрос подтверждения ---
print("\n" + "="*50)
print("Шаг 3: Подтверждение записи JSONL файла")
print(f"Будет создан файл '{output_filename}' в директории '{output_directory}'.")
print(f"В файл будет записано {len(selected_articles_cleaned_texts)} отобранных статей.")

user_confirmation = input("Продолжить запись в файл? (y/n): ").strip().lower()

if user_confirmation != 'y':
    print("\nЗапись файла отменена пользователем. Скрипт завершен.")
    sys.exit(0) # Завершаем выполнение без ошибки

# --- Шаг 4: Запись отобранных статей в JSON Lines файл ---
print("\n" + "="*50)
print("Шаг 4: Запись отобранных статей в JSON Lines файл...")

output_full_path = os.path.join(output_directory, output_filename)

try:
    # Создаем директорию для выходного файла, если она не существует
    os.makedirs(output_directory, exist_ok=True)
    print(f"Директория '{output_directory}' готова.")

    # Открываем файл для записи
    with open(output_full_path, 'w', encoding='utf-8') as f:
        # Записываем каждую отобранную статью как отдельный JSON-объект
        for i, cleaned_text in enumerate(selected_articles_cleaned_texts):
            # Формируем JSON-объект согласно требованиям
            article_json = {
                "number": i, # Порядковый номер в этом файле
                "text": cleaned_text # Очищенный текст
                # Другие поля не сохраняем, как было запрошено
            }
            # Преобразуем словарь в JSON строку, ensure_ascii=False сохраняет русские символы
            json_line = json.dumps(article_json, ensure_ascii=False)
            # Записываем строку в файл, добавляя перенос строки
            f.write(json_line + '\n')

            # Выводим прогресс записи
            if (i + 1) % 1000 == 0:
                print(f"    Записано {i + 1}/{len(selected_articles_cleaned_texts)} статей в файл...")

    print(f"\nШаг 4 завершен. Файл '{output_full_path}' успешно создан.")
    print(f"Всего записано {len(selected_articles_cleaned_texts)} статей.")

except Exception as e:
    print(f"\nКритическая ошибка во время Шага 4 (запись файла): {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nСкрипт успешно завершил работу.")
print("="*50)