import requests
import json
import time
import os # Для работы с файловой системой

# --- НАСТРОЙКА LM STUDIO API ---
API_URL = "http://localhost:1234/v1/chat/completions" # Порт LM Studio
HEADERS = {
    "Content-Type": "application/json",
}
# В LM Studio "model" в payload часто игнорируется
MODEL_NAME_PAYLOAD = "gpt-3.5-turbo" # Или любое другое название

# --- НАСТРОЙКИ ФАЙЛОВ И ПАПОК ---
# Папка, содержащая JSON файлы с заголовками и описаниями
INPUT_JSON_DIR = "titles_chunk1" # Папка из предыдущего скрипта

# Папка, куда будут сохраняться сгенерированные статьи в формате JSONL
OUTPUT_JSONL_DIR = "articles_chunk1"

# --- НАСТРОЙКИ ГЕНЕРАЦИИ ---
DELAY_BETWEEN_REQUESTS = 1 # Задержка между запросами (сек), чтобы не перегружать LM Studio

# Шаблон промта для генерации статьи (используем {title} И {description})
PROMPT_TEMPLATE = """
Напиши связный текст для энциклопедической статьи.
Заголовок статьи: «{title}»
Вот краткое описание содержания статьи для контекста: {description}

Объём — 5–6 абзацев.
Текст должен быть полностью связанным, без списков, без подзаголовков, без маркированных или нумерованных пунктов, без ссылок на источники, без примечаний, без вставок типа "См. также" или "Источники".
Пиши нормальным литературным стилем, как в энциклопедии для широкой аудитории.
Не используй разметку, скобочные ссылки и другие формальности Википедии.
Просто цельный, аккуратный, логичный текст на русском языке.
""".strip() # .strip() убирает лишние пустые строки в начале и конце шаблона

# --- ФУНКЦИИ ---

def generate_article_text(title: str, description: str) -> str | None:
    """
    Генерирует текст статьи с помощью LM Studio API по заголовку и описанию.
    """
    prompt = PROMPT_TEMPLATE.format(title=title, description=description)
    payload = {
        "model": MODEL_NAME_PAYLOAD,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # "temperature": 0.7, # Можно настроить креативность модели (от 0 до 2)
        # "max_tokens": 2500, # Установите достаточное количество токенов для статьи (~5-6 абзацев может потребовать 1500-2500 токенов)
        # "stream": False # Обычно False для одноразового ответа
    }
    try:
        # print(f"Отправка промпта для статьи: «{title}»") # Отладочный вывод
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status() # Вызовет исключение HTTPError для плохих ответов (4xx or 5xx)

        response_data = response.json()
        # Извлекаем текст из стандартного OpenAI-совместимого формата ответа
        if 'choices' in response_data and len(response_data['choices']) > 0 and 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
            text = response_data['choices'][0]['message']['content'].strip()
            # print("Текст статьи успешно получен.") # Отладочный вывод
            return text
        else:
            print(f"Предупреждение: Неожиданная структура ответа LM Studio API для '{title}'.")
            print(f"API ответ: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса к LM Studio API при генерации '{title}': {e}")
        if hasattr(response, 'text'):
             print(f"API ответ (если есть): {response.text}")
        return None
    except Exception as e:
        print(f"Непредвиденная ошибка при генерации текста для '{title}': {e}")
        return None


def process_json_file(input_filepath: str, output_filepath: str, delay: float, api_url: str, headers: dict, model_name_payload: str):
    """
    Обрабатывает один входной JSON файл, генерирует статьи и сохраняет в один JSONL файл.
    """
    items_to_process = []
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            items_to_process = json.load(f)
        if not isinstance(items_to_process, list):
             print(f"Ошибка: Файл '{input_filepath}' не содержит JSON массива верхнего уровня. Пропускаем.")
             return
        print(f"Прочитано {len(items_to_process)} элементов из '{input_filepath}'.")
    except FileNotFoundError:
        print(f"Ошибка: Входной файл '{input_filepath}' не найден. Пропускаем.")
        return
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON в файле '{input_filepath}': {e}. Пропускаем.")
        return
    except Exception as e:
        print(f"Непредвиденная ошибка при чтении файла '{input_filepath}': {e}. Пропускаем.")
        return

    if not items_to_process:
        print(f"В файле '{input_filepath}' нет элементов для обработки.")
        return

    # Открываем файл на запись JSONL
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            total_items = len(items_to_process)
            generated_count = 0
            for idx, item in enumerate(items_to_process, 1):
                # Извлекаем данные из элемента JSON
                item_number = item.get('number', idx) # Используем номер из JSON или порядковый номер, если ключа нет
                title = item.get('title')
                description = item.get('description')

                if not title or not description:
                    print(f"Предупреждение: Элемент {item_number} в '{input_filepath}' не содержит 'title' или 'description'. Пропускаем.")
                    continue # Пропускаем этот элемент

                print(f"[{idx}/{total_items}] (№{item_number}) Генерация текста для статьи: «{title}»")

                # Генерируем текст статьи
                article_text = generate_article_text(title, description)

                if article_text:
                    # Формируем объект для сохранения в JSONL
                    output_item = {
                        "number": item_number,
                        "title": title,
                        "description": description,
                        "text": article_text
                    }
                    # Записываем объект как строку JSON с новой строки
                    f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                    generated_count += 1
                else:
                    print(f"Не удалось сгенерировать текст для статьи «{title}».")

                # Пауза между запросами
                if idx < total_items: # Пауза после всех, кроме последнего
                     time.sleep(delay)

        print(f"Обработка файла '{input_filepath}' завершена. Успешно сгенерировано {generated_count} статей, сохранено в '{output_filepath}'.")

    except Exception as e:
        print(f"Ошибка при записи в выходной файл '{output_filepath}': {e}")


def main():
    """
    Основная логика скрипта: сканирует входную папку, обрабатывает каждый JSON файл.
    """
    # Создаем выходную папку, если она не существует
    os.makedirs(OUTPUT_JSONL_DIR, exist_ok=True)
    print(f"Папка для сохранения статей (JSONL): '{OUTPUT_JSONL_DIR}' (создана или уже существует).")

    # Сканируем входную папку на наличие JSON файлов
    input_files = [f for f in os.listdir(INPUT_JSON_DIR) if f.endswith('.json')]

    if not input_files:
        print(f"Входная папка '{INPUT_JSON_DIR}' не содержит файлов с расширением .json. Скрипт завершен.")
        return

    print(f"Найдено {len(input_files)} JSON файлов для обработки в папке '{INPUT_JSON_DIR}'.")

    # Обрабатываем каждый найденный JSON файл
    for i, filename in enumerate(input_files, 1):
        input_filepath = os.path.join(INPUT_JSON_DIR, filename)
        # Формируем имя выходного JSONL файла (заменяем .json на .jsonl)
        output_filename = filename.replace('.json', '.jsonl')
        output_filepath = os.path.join(OUTPUT_JSONL_DIR, output_filename)

        print(f"\n--- Обработка файла {i}/{len(input_files)}: '{filename}' ---")

        # Вызываем функцию для обработки отдельного файла
        process_json_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            delay=DELAY_BETWEEN_REQUESTS,
            api_url=API_URL,
            headers=HEADERS,
            model_name_payload=MODEL_NAME_PAYLOAD
        )
        print("-" * 30)


if __name__ == "__main__":
    # Убедитесь, что LM Studio запущен, модель загружена и API сервер запущен
    # Убедитесь, что у вас есть папка INPUT_JSON_DIR с JSON файлами, сгенерированными предыдущим скриптом.
    main()