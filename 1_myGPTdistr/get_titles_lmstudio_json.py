import os
import time
import re
import json
import requests # Импортируем библиотеку для HTTP-запросов

# --- НАСТРОЙКА ---
# URL локального API, который предоставляет LM Studio
# Убедитесь, что порт соответствует настройкам LM Studio (обычно 1234)
API_URL = "http://localhost:1234/v1/chat/completions"

# Имя файла с темами
THEMES_FILE = "themes/1.txt"

# Имя папки, куда будут сохраняться сгенерированные заголовки и описания (в JSON)
OUTPUT_DIR = "generated_content_local" # Изменим название папки, чтобы не путать с Gemini

# Имя подпапки для сохранения ответов, которые не удалось распарсить (в TXT)
FAILED_RESPONSES_DIR_NAME = "failed_json_responses"

# Количество пар "номер" + "заголовок" + "описание" для генерации по каждой теме
NUMBER_OF_ITEMS = 20

# Задержка между запросами к API (в секундах).
# Для локальной модели можно поставить 0 или очень маленькое значение,
# т.к. нет ограничений RPM, кроме производительности вашего железа.
REQUEST_DELAY = 1 # 1 секунда (или меньше, если модель очень быстрая)

# Настройки для запроса к LM Studio API (обычно не требуют ключа)
HEADERS = {
    "Content-Type": "application/json",
}
# В LM Studio "model" в payload часто игнорируется, используется загруженная модель.
# Можно указать что-то для совместимости или для явного указания, если LM Studio это поддерживает.
# MODEL_NAME_PAYLOAD = "gpt-3.5-turbo" # Пример
MODEL_NAME_PAYLOAD = "gpt-3.5-turbo" # Или такое название

# --- ФУНКЦИИ ---

def generate_prompt(title: str, number: int) -> str:
    """
    Формирует промпт для языковой модели, запрашивая JSON с номером, заголовком и описанием.
    (Эта функция остается почти без изменений, только убираем пример JSON для краткости промпта)
    """
    prompt_template = """
Сгенерируй ровно {number} объектов в формате **чистого JSON массива**.
**Не включай никаких дополнительных символов, комментариев или блоков Markdown** (например, ```json) до или после JSON массива.

Каждый объект в массиве должен содержать следующие ключи:
1.  "number" (целое число): Порядковый номер элемента в списке, начиная с 1 и до {number}.
2.  "title" (строка): Краткий и содержательный заголовок для энциклопедической статьи по теме «{title}». Стиль: научный или популярно-научный.
3.  "description" (строка): Краткое описание статьи (1-3 предложения), поясняющее суть заголовка для маленькой локальной модели.

Сгенерируй ровно {number} объектов для темы «{title}».
"""
    return prompt_template.format(title=title, number=number)

def read_themes_from_file(filename: str) -> list[str]:
    """Считывает темы из текстового файла."""
    themes = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                theme = line.strip()
                if theme:
                    themes.append(theme)
        print(f"Считано {len(themes)} тем из файла '{filename}'.")
    except FileNotFoundError:
        print(f"Ошибка: Файл тем '{filename}' не найден.")
    except Exception as e:
        print(f"Ошибка при чтении файла '{filename}': {e}")
    return themes

def sanitize_filename(theme_title: str, extension: str = ".json") -> str:
    """Очищает строку темы для использования в качестве имени файла."""
    filename = theme_title.replace(' ', '_')
    filename = re.sub(r'[^\w_-]', '', filename)
    filename = filename.strip('_-')
    if len(filename) > 100:
        filename = filename[:100].rstrip('_-')
    if not filename:
         filename = "untitled_theme_" + str(int(time.time()))
    return filename + extension

def save_content_to_json_file(output_dir: str, theme_title: str, content: list[dict]):
    """Сохраняет сгенерированный список объектов в JSON файл."""
    filename = sanitize_filename(theme_title, extension=".json")
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
        print(f"-> Успешный результат сохранен в файл: '{filepath}'")
    except Exception as e:
        print(f"Ошибка при сохранении JSON файла '{filepath}': {e}")

def save_failed_response(output_dir: str, theme_title: str, content: str):
    """Сохраняет сырой текст ответа модели в TXT файл в подпапке для ошибок."""
    timestamp = int(time.time())
    base_filename = sanitize_filename(theme_title, extension="").replace(".json", "")
    filename = f"{base_filename}_{timestamp}_failed.txt" # Добавим _failed для ясности
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"-> Сырой ответ сохранен в файл (не удалось распарсить JSON): '{filepath}'")
    except Exception as e:
        print(f"Ошибка при сохранении сырого ответа в файл '{filepath}': {e}")

def extract_json_from_text(text: str) -> str | None:
    """Пытается найти и извлечь строку, содержащую JSON, из текста."""
    if not text:
        return None

    first_brace_index = -1
    for char in ['[', '{']:
        index = text.find(char)
        if index != -1:
            if first_brace_index == -1 or index < first_brace_index:
                first_brace_index = index

    last_brace_index = -1
    for char in [']', '}']:
        index = text.rfind(char)
        if index != -1:
             if last_brace_index == -1 or index > last_brace_index:
                 last_brace_index = index

    if first_brace_index != -1 and last_brace_index != -1 and first_brace_index < last_brace_index:
        json_string = text[first_brace_index : last_brace_index + 1]
        return json_string.strip()
    else:
        return None


def process_themes_lmstudio(api_url: str, themes_file: str, output_dir: str, failed_dir_name: str, num_items: int, request_delay: int, headers: dict, model_name_payload: str):
    """
    Главная функция: считывает темы, генерирует промпты, отправляет их в локальный API
    LM Studio, парсит JSON, выводит количество объектов и сохраняет результаты.
    """
    # Определяем пути к папкам
    main_output_dir = output_dir
    failed_output_dir = os.path.join(main_output_dir, failed_dir_name)

    # Создаем папки, если они не существуют
    os.makedirs(main_output_dir, exist_ok=True)
    os.makedirs(failed_output_dir, exist_ok=True)
    print(f"Папка для успешных результатов (JSON): '{main_output_dir}' (создана или уже существует).")
    print(f"Папка для нераспарсенных ответов (TXT): '{failed_output_dir}' (создана или уже существует).")


    # Считываем темы
    themes = read_themes_from_file(themes_file)
    if not themes:
        print("Нет тем для обработки. Скрипт завершен.")
        return

    print(f"API URL: {api_url}")
    print(f"Ожидается генерация {num_items} пар 'номер' + 'заголовок' + 'описание' для каждой темы.")

    # Обработка каждой темы
    total_themes = len(themes)
    for i, theme in enumerate(themes):
        print(f"\n--- Обработка темы {i+1}/{total_themes}: «{theme}» ---")

        prompt = generate_prompt(theme, num_items)

        # Подготовка payload для запроса к LM Studio API
        payload = {
            "model": model_name_payload,
            "messages": [
                {"role": "user", "content": prompt}
            ],
             # "temperature": 0.7, # Можно настроить креативность модели
             # "max_tokens": 2000, # Установите достаточное количество токенов для ответа
             # "stream": False # Обычно False для одноразового ответа
        }


        generated_content = None
        raw_response_text = None
        response_successful = False # Флаг для отслеживания успешности получения текста ответа от модели

        try:
            print(f"Отправка запроса к локальному API для темы «{theme}»...")
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Вызовет исключение для кодов ответа 4xx/5xx

            response_successful = True # API запрос успешен (Status 200)
            print("API запрос успешен (Status 200). Попытка получить текст ответа модели...")

            # Парсим ответ от LM Studio API (OpenAI-совместимый формат)
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0 and 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                 raw_response_text = response_data['choices'][0]['message']['content']
                 if raw_response_text:
                    raw_response_text = raw_response_text.strip()
                    print("Текст ответа модели получен. Попытка извлечь и распарсить JSON...")
                    # Продолжаем с существующей логикой парсинга JSON
                    json_string_to_parse = extract_json_from_text(raw_response_text)

                    if json_string_to_parse:
                        try:
                            generated_content = json.loads(json_string_to_parse)
                            print("JSON успешно распарсен.")
                            # --- ВЫВОД КОЛИЧЕСТВА ОБЪЕКТОВ ---
                            if isinstance(generated_content, list):
                                print(f"-> Получено {len(generated_content)} объектов (номер + заголовок + описание). Ожидалось: {num_items}.")
                            # --- КОНЕЦ ВЫВОДА ---

                        except json.JSONDecodeError as e:
                            print(f"Ошибка парсинга JSON для темы «{theme}»: {e}")
                            print("Часть ответа, которую пытались распарсить как JSON:")
                            print(json_string_to_parse)
                            # Сырой ответ будет сохранен ниже
                            generated_content = None
                        except Exception as e:
                             print(f"Неизвестная ошибка при парсинге JSON для темы «{theme}»: {e}")
                             print("Часть ответа, которую пытались распарсить как JSON:")
                             print(json_string_to_parse)
                             # Сырой ответ будет сохранен ниже
                             generated_content = None
                    else:
                         print(f"Не удалось найти ожидаемую структуру JSON (не найдены открывающие/закрывающие скобки [] или {{}}) в ответе модели для темы «{theme}».")
                         # Сырой ответ будет сохранен ниже
                         generated_content = None
                 else:
                     print(f"Предупреждение: Модель вернула пустой текстовый ответ для темы «{theme}».")
                     # В этом случае raw_response_text будет пустым, сохранять нечего в ошибки
                     response_successful = False # Ответ пустой, считаем не совсем успешным

            else:
                print(f"Предупреждение: Структура ответа LM Studio API неожиданна для темы «{theme}». Не найден 'choices[0].message.content'.")
                print("Полученный API ответ:")
                print(response.text) # Выводим сырой ответ API для отладки
                # raw_response_text не получен, сохранять нечего в ошибки
                response_successful = False # Ответ не в ожидаемом формате, считаем неуспешным


        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса к локальному API для темы «{theme}»: {e}")
            if response is not None and hasattr(response, 'text'):
                 print(f"API ответ (если есть): {response.text}") # Выводим ответ, если он был получен до ошибки
            # raw_response_text не получен в случае ошибки запроса
            response_successful = False # Ошибка запроса, считаем неуспешным


        # --- ЛОГИКА СОХРАНЕНИЯ ---
        # Если JSON был успешно распарсен и это непустой список, сохраняем его в JSON файл
        if isinstance(generated_content, list) and generated_content:
             save_content_to_json_file(main_output_dir, theme, generated_content)
        elif isinstance(generated_content, list) and not generated_content:
             print(f"Предупреждение: Модель вернула пустой JSON массив для темы «{theme}».")
             # Пустой массив не считаем ошибкой, требующей сохранения сырого ответа

        # Если не удалось распарсить JSON И при этом был получен какой-то непустой raw_response_text
        elif raw_response_text and response_successful: # response_successful = True означает, что был 200 статус и текст content
            print(f"Сохраняем сырой ответ для темы «{theme}» в папку нераспарсенных ответов...")
            save_failed_response(failed_output_dir, theme, raw_response_text)
        # В остальных случаях (нет raw_response_text, ошибка запроса, пустой ответ) ничего не сохраняем


        # Пауза между запросами, если это не последняя тема
        if i < total_themes - 1:
            print(f"Пауза {request_delay} секунд...")
            time.sleep(request_delay)

    print("\n--- Обработка тем завершена ---")


# --- ЗАПУСК ---
if __name__ == "__main__":
    # Убедитесь, что LM Studio запущен и модель загружена,
    # и API сервер запущен (обычно на http://localhost:1234/v1)
    process_themes_lmstudio(
        api_url=API_URL,
        themes_file=THEMES_FILE,
        output_dir=OUTPUT_DIR,
        failed_dir_name=FAILED_RESPONSES_DIR_NAME,
        num_items=NUMBER_OF_ITEMS,
        request_delay=REQUEST_DELAY,
        headers=HEADERS,
        model_name_payload=MODEL_NAME_PAYLOAD # Передаем имя модели для payload
    )
