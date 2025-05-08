from datasets import load_from_disk
import os
import traceback
import ast
import re
import statistics # Импортируем библиотеку для статистики

# --- Конфигурация ---
base_save_directory = "./google_wiki40b_ru"
splits_order = ['train', 'validation', 'test']
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

    # --- Логика обработки и декодирования содержимого (повтор из get_article_text_by_global_index) ---
    article_text_decoded = "" # Инициализируем пустой строкой на случай ошибок

    if isinstance(article_content, str):
        try:
            evaluated_content = ast.literal_eval(article_content)
            if isinstance(evaluated_content, bytes):
                try:
                    article_text_decoded = evaluated_content.decode('utf-8')
                except Exception:
                    print(f"\nПредупреждение (декодирование байт): Не удалось декодировать байты в UTF-8. Содержимое: {evaluated_content[:100]}...") # Ограничим вывод
            else:
                print(f"\nПредупреждение (оценка строки): Оценка строки дала неожиданный тип ({type(evaluated_content)}) вместо байт. Содержимое: {article_content[:100]}...")
        except (ValueError, SyntaxError, TypeError):
             print(f"\nПредупреждение (оценка строки): Строка не является корректным литералом Python. Содержимое: {article_content[:100]}...")
        except Exception as e:
             print(f"\nПредупреждение (оценка строки): Непредвиденная ошибка при оценке строки: {e}. Содержимое: {article_content[:100]}...")

    elif isinstance(article_content, bytes):
        try:
            article_text_decoded = article_content.decode('utf-8')
        except Exception:
             print(f"\nПредупреждение (декодирование байт напрямую): Не удалось декодировать байты в UTF-8. Содержимое: {article_content[:100]}...")

    else:
        print(f"\nПредупреждение (неожиданный тип): Поле 'text' имеет неожиданный тип: {type(article_content)}. Содержимое: {article_content}")
    # --- Конец логики декодирования ---

    # Теперь применяем очистку к декодированному тексту (если это строка)
    if isinstance(article_text_decoded, str):
        cleaned_text = clean_wiki_text(article_text_decoded)
        return cleaned_text
    else:
        # Если на этапе декодирования/обработки произошла ошибка и результат не строка
        return "" # Возвращаем пустую строку, чтобы ее длина была 0

# --- Конец вспомогательной функции ---


# --- Часть загрузки датасетов ---
loaded_splits = {}

print(f"Загрузка датасета из локальной папки: {base_save_directory}")

try:
    for split_name in splits_order:
        split_path = os.path.join(base_save_directory, split_name)
        if os.path.exists(split_path):
            print(f"Загрузка раздела '{split_name}' из '{split_path}'...")
            try:
                loaded_splits[split_name] = load_from_disk(split_path)
                print(f"Раздел '{split_name}' успешно загружен. Статей: {loaded_splits[split_name].num_rows}")
            except Exception as e:
                print(f"Ошибка при загрузке раздела '{split_name}' из '{split_path}': {e}")
        else:
            print(f"Папка раздела '{split_path}' не найдена по пути '{split_path}'. Пропускаем загрузку раздела '{split_name}'.")

    if not loaded_splits:
        print("\nКритическая ошибка: Не удалось загрузить ни один раздел датасета.")
        print(f"Пожалуйста, убедитесь, что папка '{base_save_directory}' существует и содержит подпапки train, validation, test, сохраненные с помощью save_to_disk().")
        exit()

    print("\nВсе доступные разделы загружены в словарь 'loaded_splits'.")

except Exception as e:
    print(f"\nПроизошла непредвиденная ошибка в начале скрипта во время попытки загрузки: {e}")
    traceback.print_exc()
    exit()


# --- Анализ длины статей ---
print("\n" + "="*30)
print("Начало анализа длины очищенных статей...")

article_lengths = []
total_articles_processed = 0

# Проходим по всем загруженным разделам
for split_name in splits_order:
    if split_name in loaded_splits:
        print(f"  Обработка раздела '{split_name}'...")
        dataset_split = loaded_splits[split_name]
        num_examples_in_split = len(dataset_split) # Используем len() или .num_rows

        # Итерируемся по примерам в текущем разделе
        for i, example in enumerate(dataset_split):
            if (i + 1) % 10000 == 0: # Выводим прогресс каждые 10000 статей
                print(f"    Обработано {i + 1}/{num_examples_in_split} статей в разделе '{split_name}'")

            # Получаем и очищаем текст с помощью вспомогательной функции
            cleaned_text = get_decoded_cleaned_text(example)

            # Считаем длину очищенного текста (количество символов)
            length = len(cleaned_text)

            # Добавляем длину в список
            article_lengths.append(length)
            total_articles_processed += 1

print(f"\nАнализ завершен. Обработано {total_articles_processed} статей.")

if total_articles_processed == 0:
    print("Не найдено статей для анализа длины.")
else:
    # Сортируем длины для расчета медианы и процентов
    article_lengths.sort()

    # 1. Расчет среднего и медианы для всех статей
    overall_mean_length = statistics.mean(article_lengths)
    overall_median_length = statistics.median(article_lengths)

    print(f"\nОбщая статистика длины (в символах):")
    print(f"  Средняя длина: {overall_mean_length:.2f}")
    print(f"  Медианная длина: {overall_median_length}")

    # 2. Расчет среднего для 10% самых коротких и самых длинных
    percentile = 0.10
    k = max(1, int(percentile * total_articles_processed)) # Количество статей для расчета (минимум 1)

    if total_articles_processed >= 10: # Рассчитываем процентили только если статей достаточно
        # 10% самых коротких
        shortest_10_percent_lengths = article_lengths[:k]
        mean_shortest_10 = statistics.mean(shortest_10_percent_lengths)
        print(f"\nСтатистика для {k} ({percentile:.0%}) самых коротких статей:")
        print(f"  Средняя длина: {mean_shortest_10:.2f}")
        print(f"  Самая короткая статья: {article_lengths[0]} символов")

        # 10% самых длинных
        longest_10_percent_lengths = article_lengths[-k:]
        mean_longest_10 = statistics.mean(longest_10_percent_lengths)
        print(f"\nСтатистика для {k} ({percentile:.0%}) самых длинных статей:")
        print(f"  Средняя длина: {mean_longest_10:.2f}")
        print(f"  Самая длинная статья: {article_lengths[-1]} символов")
    else:
        print(f"\nНедостаточно статей ({total_articles_processed}) для расчета статистики по 10% самых коротких/длинных.")

print("\n" + "="*30)
print("Анализ длины завершен.")

# --- Интерактивный цикл (можно закомментировать, если нужен только анализ) ---
# ... (код интерактивного цикла из предыдущего скрипта) ...
# Чтобы использовать его, вам нужно убедиться, что total_articles, split_offsets
# и loaded_splits определены выше и остаются доступными.
# Можете просто скопировать сюда блок 'while True:' из предыдущего скрипта.
# Не забудьте убрать или изменить print(f"Тип переменной article_text после получения: {type(article_text)}")
# и использовать cleaned_text = get_decoded_cleaned_text(example)
# вместо получения и ручной очистки внутри цикла.