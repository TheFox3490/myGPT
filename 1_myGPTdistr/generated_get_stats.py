import os
import json
import statistics
import traceback

# --- Конфигурация ---
# Папка, содержащая файлы сгенерированных статей в формате .jsonl
generated_data_directory = "./generated_articles_jsonl"
# --- Конец конфигурации ---

# --- Анализ длины сгенерированных статей ---
print("="*30)
print("Начало анализа длины сгенерированных статей...")
print(f"Директория с данными: {generated_data_directory}")

generated_article_lengths = []
total_files_processed = 0
total_articles_processed = 0
error_count = 0

# Проверяем, существует ли папка с данными
if not os.path.exists(generated_data_directory):
    print(f"\nОшибка: Директория с данными '{generated_data_directory}' не найдена.")
    print("Пожалуйста, убедитесь, что папка существует и указан правильный путь.")
    exit() # Завершаем выполнение, если папка не найдена

# Получаем список всех файлов .jsonl в директории
try:
    jsonl_files = [f for f in os.listdir(generated_data_directory) if f.endswith('.jsonl')]
except Exception as e:
    print(f"\nОшибка при получении списка файлов из директории '{generated_data_directory}': {e}")
    traceback.print_exc()
    exit()

if not jsonl_files:
    print(f"\nНе найдено файлов с расширением '.jsonl' в директории '{generated_data_directory}'.")
    print("Пожалуйста, убедитесь, что файлы присутствуют и имеют правильное расширение.")
else:
    print(f"Найдено {len(jsonl_files)} файлов .jsonl.")
    
    # Итерируемся по каждому файлу .jsonl
    for file_index, file_name in enumerate(jsonl_files):
        file_path = os.path.join(generated_data_directory, file_name)
        total_files_processed += 1
        print(f"\n  Обработка файла {total_files_processed}/{len(jsonl_files)}: '{file_name}'")

        # Открываем и читаем файл построчно
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                articles_in_file = 0
                for line_number, line in enumerate(f):
                    try:
                        # Парсим каждую строку как JSON-объект
                        article_data = json.loads(line)

                        # Извлекаем текст статьи
                        # Используем .get() для безопасного доступа к ключу 'text'
                        article_text = article_data.get('text')

                        if article_text is not None:
                            # Считаем длину текста (количество символов)
                            # Предполагаем, что текст уже является строкой
                            length = len(article_text)

                            # Добавляем длину в список
                            generated_article_lengths.append(length)
                            total_articles_processed += 1
                            articles_in_file += 1
                        else:
                            # Обрабатываем случай, если ключа 'text' нет в объекте
                            print(f"    Предупреждение: В файле '{file_name}' на строке {line_number + 1} отсутствует ключ 'text'. Пропускаем эту статью.")
                            error_count += 1

                    except json.JSONDecodeError:
                        print(f"    Предупреждение: Ошибка парсинга JSON в файле '{file_name}' на строке {line_number + 1}. Пропускаем строку.")
                        error_count += 1
                    except Exception as e:
                        print(f"    Предупреждение: Непредвиденная ошибка при обработке статьи в файле '{file_name}' на строке {line_number + 1}: {e}. Пропускаем статью.")
                        error_count += 1

            print(f"  Обработка файла '{file_name}' завершена. Найдено статей: {articles_in_file}")

        except FileNotFoundError:
            print(f"  Ошибка: Файл '{file_name}' не найден. Пропускаем.")
            error_count += 1
        except Exception as e:
            print(f"  Ошибка при чтении файла '{file_name}': {e}. Пропускаем.")
            traceback.print_exc()
            error_count += 1

# --- Вывод статистики ---
print(f"\nАнализ завершен. Всего обработано файлов: {total_files_processed}. Всего найдено статей: {total_articles_processed}. Ошибок/пропусков: {error_count}.")

if total_articles_processed == 0:
    print("\nНе найдено статей для анализа длины.")
else:
    # Сортируем длины для расчета медианы и процентов
    generated_article_lengths.sort()

    # 1. Расчет среднего и медианы для всех статей
    overall_mean_length = statistics.mean(generated_article_lengths)
    overall_median_length = statistics.median(generated_article_lengths)

    print(f"\nОбщая статистика длины сгенерированных статей (в символах):")
    print(f"  Средняя длина: {overall_mean_length:.2f}")
    print(f"  Медианная длина: {overall_median_length}")
    print(f"  Самая короткая сгенерированная статья: {generated_article_lengths[0]} символов")
    print(f"  Самая длинная сгенерированная статья: {generated_article_lengths[-1]} символов")


    # 2. Расчет среднего для 10% самых коротких и самых длинных
    percentile = 0.10
    k = max(1, int(percentile * total_articles_processed)) # Количество статей для расчета (минимум 1)

    if total_articles_processed >= 10: # Рассчитываем процентили только если статей достаточно
        # 10% самых коротких
        shortest_10_percent_lengths = generated_article_lengths[:k]
        mean_shortest_10 = statistics.mean(shortest_10_percent_lengths)
        print(f"\nСтатистика для {k} ({percentile:.0%}) самых коротких сгенерированных статей:")
        print(f"  Средняя длина: {mean_shortest_10:.2f}")


        # 10% самых длинных
        longest_10_percent_lengths = generated_article_lengths[-k:]
        mean_longest_10 = statistics.mean(longest_10_percent_lengths)
        print(f"\nСтатистика для {k} ({percentile:.0%}) самых длинных сгенерированных статей:")
        print(f"  Средняя длина: {mean_longest_10:.2f}")

    else:
        print(f"\nНедостаточно статей ({total_articles_processed}) для расчета статистики по 10% самых коротких/длинных.")

print("\n" + "="*30)
print("Анализ длины сгенерированных статей завершен.")