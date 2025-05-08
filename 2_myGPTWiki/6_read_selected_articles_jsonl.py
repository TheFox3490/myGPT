import json
import os

# Заданный путь к файлу
FILE_PATH = "selected_wiki_jsonl/selected_wiki_articles.jsonl"

def count_articles(file_path):
    """
    Подсчитывает количество строк (статей) в файле.
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(file_path):
            return 0

        # Быстрый подсчет строк без загрузки в память
        with open(file_path, 'rb') as f:
            # Используем byte-операции для скорости
            # Читаем по чанкам и считаем символы перевода строки
            count = 0
            buf_size = 1024 * 1024 # 1 MB
            buf = f.raw.read(buf_size)
            while buf:
                count += buf.count(b'\n')
                buf = f.raw.read(buf_size)
            # Если файл не пустой и не заканчивается переводом строки, добавляем 1
            f.seek(0)
            if f.read(1) and not buf: # Проверка, что файл не пустой
                 f.seek(-1, os.SEEK_END)
                 if f.read(1) != b'\n':
                      count += 1
            return count

    except Exception as e:
        print(f"Произошла ошибка при подсчете статей в файле: {e}")
        return 0

def read_article_text_by_index(file_path, index):
    """
    Читает JSON Lines файл, извлекает и выводит текст статьи по индексу.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == index:
                    try:
                        article_data = json.loads(line)
                        # Извлекаем текст по ключу 'text'
                        article_text = article_data.get('text') # Используем .get() для безопасного доступа

                        if article_text is not None:
                            print(f"--- Текст статьи с индексом {index} ---")
                            print(article_text)
                            print("-------------------------------------")
                        else:
                            print(f"Ошибка: В JSON объекте по индексу {index} отсутствует ключ 'text' или его значение равно None.")

                    except json.JSONDecodeError:
                        print(f"Ошибка: Не удалось разобрать JSON строку по индексу {index}.")
                    except Exception as e:
                        print(f"Произошла ошибка при обработке JSON объекта по индексу {index}: {e}")
                    return

            # Если цикл завершился, значит, индекс не найден
            print(f"Ошибка: Статья с индексом {index} не найдена. Возможно, индекс вне диапазона (0 - {count_articles(file_path) - 1}).")

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути: {file_path}")
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")


if __name__ == "__main__":
    print(f"Попытка загрузить данные из файла: {FILE_PATH}")

    total_articles = count_articles(FILE_PATH)

    if total_articles == 0:
        if not os.path.exists(FILE_PATH):
             print("Файл не найден.")
        else:
             print("Файл пуст или произошла ошибка при его чтении/подсчете строк.")
        print("Программа завершена.")
        sys.exit() # Выходим, если файл не найден или пуст

    print(f"Всего статей в файле: {total_articles}")
    print(f"Доступные индексы: от 0 до {total_articles - 1}")
    print("Данные готовы к просмотру.")

    while True:
        try:
            index_input = input(f"Введите индекс статьи (от 0 до {total_articles - 1}), 'q' для выхода: ")
            if index_input.lower() == 'q':
                break

            article_index = int(index_input)

            if 0 <= article_index < total_articles:
                 read_article_text_by_index(FILE_PATH, article_index)
            else:
                 print(f"Неверный индекс. Пожалуйста, введите число от 0 до {total_articles - 1}.")

        except ValueError:
            print("Неверный ввод. Пожалуйста, введите целое число для индекса или 'q' для выхода.")
        except Exception as e:
            print(f"Произошла непредвиденная ошибка: {e}")

    print("Программа завершена.")