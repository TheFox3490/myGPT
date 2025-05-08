import json
import os

def build_article_index(filepath):
    """
    Проходит по JSONL файлу и записывает байтовые смещения начала каждой строки.
    Возвращает общее количество строк/статей и список смещений.
    """
    print(f"Построение индекса для файла: {filepath}...")
    offsets = []
    article_count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            while True:
                offset = f.tell() # Запоминаем текущее положение в файле
                line = f.readline()
                if not line: # Конец файла
                    break
                # Можно добавить минимальную проверку на валидность JSON здесь,
                # но для построения индекса достаточно просто записать смещение строки.
                # Полная проверка и парсинг будут при запросе конкретной статьи.
                offsets.append(offset)
                article_count += 1
        print(f"Индекс успешно построен. Найдено {article_count} строк.")
        return article_count, offsets
    except FileNotFoundError:
        print(f"Ошибка при построении индекса: Файл не найден по пути {filepath}")
        return 0, None
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при построении индекса: {e}")
        return 0, None

def get_article_by_index(filepath, offsets, index):
    """
    Извлекает и парсит статью по ее индексу, используя список смещений.
    """
    if offsets is None or not 0 <= index < len(offsets):
        print(f"Ошибка: Некорректный индекс {index} или индекс не загружен.")
        return None

    offset = offsets[index]

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.seek(offset) # Перемещаемся к началу нужной строки
            line = f.readline() # Читаем только эту строку

            try:
                article = json.loads(line)
                 # Опционально: проверяем наличие нужных ключей при извлечении
                if 'text' in article and 'number' in article:
                    return article
                else:
                     print(f"Предупреждение: Статья по индексу {index} не содержит ключ 'text' или 'number'.")
                     return None # Возвращаем None, если статья неполная
            except json.JSONDecodeError:
                print(f"Предупреждение: Некорректная JSON строка по индексу {index}.")
                return None # Возвращаем None для некорректных строк
    except FileNotFoundError:
        print(f"Ошибка при чтении файла: Файл не найден по пути {filepath}")
        return None
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при извлечении статьи: {e}")
        return None

# --- Главная часть скрипта ---

# Запрашиваем у пользователя имя файла
file_to_inspect = input("Введите имя JSONL файла для проверки (например, random_wiki_sample_10000.jsonl): ").strip()

# Строим индекс
num_articles, article_offsets = build_article_index(file_to_inspect)

# Если индекс не удалось построить, выходим
if article_offsets is None or num_articles == 0:
    print("Не удалось загрузить или найти статьи в файле. Завершение работы.")
    exit()

# Выводим общее количество статей (по количеству строк)
print(f"\nВ файле '{file_to_inspect}' найдено {num_articles} строк (потенциальных статей).")

# Инструкции для пользователя
print("Вы можете ввести индекс статьи (от 0 до", num_articles - 1, ") чтобы увидеть ее текст.")
print("Введите 'q' для выхода.")

# Основной цикл для запроса индексов у пользователя
while True:
    user_input = input(f"Введите индекс статьи (0-{num_articles-1}) или 'q' для выхода: ").strip()

    # Проверяем, хочет ли пользователь выйти
    if user_input.lower() == 'q':
        break

    try:
        # Преобразуем ввод пользователя в целое число (индекс)
        index = int(user_input)

        # Проверяем, находится ли индекс в допустимом диапазоне
        if 0 <= index < num_articles:
            # Извлекаем статью по индексу
            article = get_article_by_index(file_to_inspect, article_offsets, index)

            if article:
                print("-" * 50) # Разделительная линия для удобства
                # Выводим номер статьи (из ключа 'number') и ее текст
                print(f"Статья с индексом {index} (Номер статьи в исходном датасете: {article.get('number', 'N/A')}):")
                print(article.get('text', 'Текст статьи отсутствует.'))
                print("-" * 50) # Разделительная линия
            # else: get_article_by_index уже вывело сообщение об ошибке/предупреждение

        else:
            print(f"Ошибка: Индекс {index} вне допустимого диапазона (0-{num_articles-1}).")

    except ValueError:
        # Обрабатываем случай, когда ввод пользователя не является числом
        print("Ошибка: Некорректный ввод. Пожалуйста, введите числовой индекс или 'q'.")
    except Exception as e:
        # Обрабатываем любые другие неожиданные ошибки
        print(f"Произошла неожиданная ошибка при обработке запроса: {e}")

# Сообщение о завершении работы
print("Выход из программы.")