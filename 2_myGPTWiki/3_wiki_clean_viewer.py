from datasets import load_from_disk
import os
import traceback
import ast # Импортируем модуль ast для оценки строкового литерала

# --- Конфигурация ---
base_save_directory = "./google_wiki40b_ru"
splits_order = ['train', 'validation', 'test']
preview_length = 1500
# --- Конец конфигурации ---


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


# --- Код, который выполняется ТОЛЬКО после УСПЕШНОЙ загрузки хотя бы одного раздела ---

# 1. Подсчет общего количества статей и расчет смещений
total_articles = 0
split_offsets = {}

print("\nПодсчет общего количества статей в загруженных разделах:")
for split_name in splits_order:
    if split_name in loaded_splits:
        num_rows = loaded_splits[split_name].num_rows
        split_offsets[split_name] = total_articles
        total_articles += num_rows
        print(f"  Раздел '{split_name}': {num_rows} статей (смещение: {split_offsets[split_name]})")

if total_articles == 0:
     print("\nОшибка: Общее количество загруженных статей равно 0. Невозможно продолжить работу с данными.")
     exit()

print(f"\nОбщее количество статей во всех загруженных разделах: {total_articles}")
print(f"Допустимый диапазон номеров статей: от 0 до {total_articles - 1}")
print("\nВведите номер статьи для просмотра ее текста или 'выход' для завершения.")


# 2. Функция для получения текста по общему номеру
def get_article_text_by_global_index(global_index, loaded_datasets_dict, split_offsets, splits_order, total_articles):
    """
    Возвращает текст статьи по ее общему индексу во всех загруженных разделах.
    """
    if not (0 <= global_index < total_articles):
        return None

    for split_name in splits_order:
        if split_name in loaded_datasets_dict:
            start_index = split_offsets[split_name]
            num_rows = loaded_datasets_dict[split_name].num_rows
            end_index = start_index + num_rows

            if start_index <= global_index < end_index:
                index_in_split = global_index - start_index
                try:
                    article = loaded_datasets_dict[split_name][index_in_split]
                    article_content = article.get('text') # Получаем значение поля 'text'

                    # --- ИЗМЕНЕНИЕ: Логика обработки строки-репрезентации байтов ---
                    article_text_processed = None # Переменная для конечной строки

                    if isinstance(article_content, str):
                        # Если содержимое является строкой (как показал тип и файл)
                        try:
                            # Попытаемся оценить строку как Python литерал
                            evaluated_content = ast.literal_eval(article_content)

                            if isinstance(evaluated_content, bytes):
                                # Если результатом оценки оказались байты, декодируем их
                                try:
                                    article_text_processed = evaluated_content.decode('utf-8')
                                except Exception:
                                     article_text_processed = f"[Ошибка: Не удалось декодировать извлеченные байты в UTF-8] {evaluated_content}"
                            else:
                                # Если оценка строки дала что-то другое, не байты
                                article_text_processed = f"[Предупреждение: Оценка строки дала неожиданный тип ({type(evaluated_content)}) вместо байт] {article_content}"

                        except (ValueError, SyntaxError, TypeError):
                            # Если строка не является корректным литералом Python (или не байт-литеролом)
                             article_text_processed = f"[Ошибка: Строка не является корректным байт-литеролом или другим простым литералом] {article_content}"
                        except Exception as e:
                             # Любые другие ошибки при оценке
                             article_text_processed = f"[Ошибка при оценке строки: {e}] {article_content}"

                    elif isinstance(article_content, bytes):
                        # Если содержимое является байтами (менее вероятно теперь, но на всякий случай)
                        try:
                            article_text_processed = article_content.decode('utf-8')
                        except Exception:
                             article_text_processed = f"[Ошибка: Не удалось декодировать байты в UTF-8] {article_content}"
                    else:
                        # Если тип данных неожиданный (ни байты, ни строка)
                        article_text_processed = f"[Ошибка: Неожиданный тип данных для текста: {type(article_content)}] {article_content}"

                    # Убедимся, что возвращаем строку, даже в случае ошибки
                    if not isinstance(article_text_processed, str):
                         article_text_processed = str(article_text_processed)

                    return (split_name, index_in_split, article_text_processed)

                except IndexError:
                     print(f"\nПредупреждение: Ошибка индексации при попытке получить статью {global_index} из раздела '{split_name}' по индексу {index_in_split}. Проверьте логику смещений.")
                     return None
                except Exception as e:
                     print(f"\nПредупреждение: Произошла ошибка при доступе к статье {global_index} ({split_name}/{index_in_split}): {e}")
                     return None

    return None


# Импортируем модуль re для регулярных выражений
import re

# --- Функция для очистки текста (вставьте сюда код функции clean_wiki_text) ---
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

# 3. Бесконечный цикл для интерактивного просмотра
print("-" * 30)
while True:
    try:
        user_input = input(f"\nВведите номер статьи (0 - {total_articles - 1}) или 'выход': ").strip().lower()

        if user_input in ['выход', 'exit', 'quit']:
            break

        article_number = int(user_input)

        result = get_article_text_by_global_index(
            article_number,
            loaded_splits,
            split_offsets,
            splits_order,
            total_articles
        )

        if result:
            split_name, index_in_split, article_text = result

            # --- ОТЛАДОЧНАЯ ПЕЧАТЬ ТИПА (оставляем для контроля) ---
            print(f"Тип переменной article_text после получения: {type(article_text)}")
            # --- КОНЕЦ ОТЛАДОЧНОЙ ПЕЧАТИ ---

            # --- Шаг очистки текста ---
            if isinstance(article_text, str): # Убеждаемся, что это строка перед очисткой
                cleaned_text = clean_wiki_text(article_text)
                print("-> Текст статьи успешно очищен от маркеров.")
            else:
                cleaned_text = "Невозможно очистить текст: данные не являются строкой."
                print(f"-> Ошибка очистки: Ожидалась строка, получен тип {type(article_text)}.")
            # --- Конец шага очистки текста ---


            # --- Сохранение в файл (сохраняем ОЧИЩЕННЫЙ текст) ---
            try:
                temp_file_dir = os.path.join(base_save_directory, "previews_clean")
                os.makedirs(temp_file_dir, exist_ok=True)
                temp_file_path = os.path.join(temp_file_dir, f"article_{article_number}_preview.txt")

                if isinstance(cleaned_text, str): # Убеждаемся, что сохраняем именно строку (очищенную)
                    with open(temp_file_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)
                    print(f"-> Очищенный текст статьи #{article_number} сохранен в файл: {temp_file_path}")
                else:
                     print(f"-> Не могу сохранить очищенный текст в файл, так как его тип неожиданный ({type(cleaned_text)}). Содержимое: {cleaned_text}")

            except Exception as file_save_error:
                print(f"-> Произошла ошибка при сохранении файла '{temp_file_path}': {file_save_error}")
            # --- Конец сохранения в файл ---


            # Вывод статьи в консоль (выводим ОЧИЩЕННЫЙ текст с предпросмотром)
            print(f"\n--- Статья #{article_number} (Раздел: '{split_name}', Индекс в разделе: {index_in_split}) ---")
            if isinstance(cleaned_text, str): # Проверяем, что очищенный текст - строка перед выводом
                if len(cleaned_text) > preview_length:
                     print(cleaned_text[:preview_length] + "...")
                     print(f"\n[Текст статьи обрезан до первых {preview_length} символов. Полный очищенный текст имеет длину {len(cleaned_text)} символов.]")
                else:
                     print(cleaned_text)
            else:
                 # Если по какой-то причине очищенный текст не строка
                 print(f"Невозможно отобразить очищенный текст: {cleaned_text}")

            print("--- Конец статьи ---")

        else:
            print(f"Ошибка: Номер статьи {article_number} вне допустимого диапазона (0 - {total_articles - 1}).")

    except ValueError:
        print("Некорректный ввод. Пожалуйста, введите целое число или 'выход'.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка в интерактивном цикле: {e}")
        # traceback.print_exc()

print("\nПрограмма завершена.")