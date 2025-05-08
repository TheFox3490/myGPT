import json
import random
import os

# Входной файл с полным датасетом статей
input_filepath = 'selected_wiki_jsonl/selected_wiki_articles.jsonl'
# Выходной файл для случайной выборки
output_filepath = 'random_wiki_sample_10000.jsonl'
# Желаемое количество статей в выборке
sample_size = 10000

print(f"Чтение статей из: {input_filepath}")
print(f"Выбор случайной выборки размером: {sample_size}")

# Инициализируем резервуар для хранения выборки
# На первых шагах просто заполняем его первыми элементами
reservoir = []
total_articles_read = 0

try:
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            total_articles_read = i + 1
            try:
                article = json.loads(line)
                if len(reservoir) < sample_size:
                    # Заполняем резервуар первыми sample_size элементами
                    reservoir.append(article)
                else:
                    # Начиная с (sample_size + 1)-го элемента,
                    # заменяем случайный элемент в резервуаре
                    # с вероятностью sample_size / total_articles_read
                    j = random.randint(0, total_articles_read - 1)
                    if j < sample_size:
                        reservoir[j] = article

            except json.JSONDecodeError:
                print(f"Предупреждение: Пропущена некорректная JSON строка на линии {total_articles_read}")
                continue # Пропускаем некорректную строку

    print(f"Всего прочитано статей: {total_articles_read}")

    # Теперь записываем статьи из резервуара в новый файл
    print(f"Запись {len(reservoir)} случайных статей в: {output_filepath}")

    # Создаем директорию для выходного файла, если она не существует
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория: {output_dir}")

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for article in reservoir:
            # Каждая статья записывается как отдельная JSON строка (JSON Lines формат)
            outfile.write(json.dumps(article, ensure_ascii=False) + '\n')

    print("Скрипт завершен. Случайная выборка сохранена.")

except FileNotFoundError:
    print(f"Ошибка: Входной файл не найден по пути {input_filepath}")
except Exception as e:
    print(f"Произошла ошибка: {e}")