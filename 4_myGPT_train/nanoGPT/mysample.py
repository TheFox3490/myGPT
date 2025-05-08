# -*- coding: utf-8 -*-

"""
Sample from a trained model using our custom dataset and tokenizer
"""
import os
import json # Импортируем json для работы с meta.json
from contextlib import nullcontext
import torch
# import tiktoken # Удаляем tiktoken
from transformers import AutoTokenizer # Импортируем AutoTokenizer
import traceback # Импортируем traceback для вывода полного стека ошибок при загрузке токенизатора

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # Всегда возобновляем с чекпойнта
# Указываем директорию, где сохранен чекпойнт вашей модели
out_dir = 'out-custom-long' # << УБЕДИТЕСЬ, ЧТО ЭТО СОВПАДАЕТ С out_dir ИЗ ВАШЕГО train_custom_corpus_small.py

# Начальная затравка для генерации.
# Установите start = "" (пустая строка), чтобы начать генерацию с токена <bos> (новая статья).
# Или укажите строку текста, чтобы продолжить с нее.
# Можно также указать файл, используя "FILE:путь/к/файлу.txt"
#start = "Нейронная сеть (также искусственная нейронная сеть, ИНС, или просто нейросеть) — математическая модель, а также её программное" # << Установите здесь вашу затравку или "" для новой статьи

#start = "Вторая мировая война(1 сентября 1939 — 2 сентября 1945) — война двух мировых военно-политических коалиций, ставшая крупнейшим вооружённым конфликтом в истории человечества.\n\nДействующие лица"

start = "Пушкин, Александр Сергеевич\n\nБиография"

#start = "Квантовая физика — раздел теоретической физики, в котором изучаются квантово-механические и квантово-полевые системы и законы их движения. Основные законы "

#start = "Искусственный интеллект (англ. artificial intelligence) в самом широком смысле — это интеллект, демонстрируемый машинами, в частности компьютерными системами. Это область исследований в области компьютерных наук, которая разрабатывает и изучает методы и программное обеспечение, позволяющие"

#start = "Гринпульки - это неизвестные науке инопланетные технологии"

num_samples = 5 # Количество примеров для генерации
max_new_tokens = 500 # Максимальное количество новых токенов в каждом примере
temperature = 0.8 # Температура генерации (0.0 - детерминированно, 1.0 - более случайное)
top_k = 200 # Учитывать только top_k наиболее вероятных токенов

seed = 1337 # Случайное зерно для воспроизводимости
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# Загружаем конфигурацию, она может переопределить параметры выше (например, из config/train_custom_corpus_small.py)
# Важно, чтобы out_dir здесь загрузился правильно из вашего файла конфига тренировки
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- Загрузка модели ---
print(f"Загрузка модели из директории {out_dir}...")
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"Ошибка: Чекпойнт модели не найден по пути: {ckpt_path}")
        print("Убедитесь, что директория out_dir в вашем конфиге sample.py (или в командной строке)")
        print("совпадает с директорией, куда сохранялись результаты тренировки.")
        exit(1)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args']) # Загружаем параметры модели из чекпойнта
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model (не наш случай)
    print(f"Инициализация из GPT-2 весов: {init_from} (не стандартный режим для нашего корпуса)")
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    print(f"Неизвестный режим инициализации модели: {init_from}. Ожидается 'resume' или 'gpt2*'.")
    exit(1)


model.eval() # Переводим модель в режим инференса
model.to(device)
if compile:
    print("Компиляция модели... (может занять некоторое время)")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
print("Модель загружена и готова к сэмплированию.")


# --- Загрузка метаданных и токенизатора ---
# Определяем путь к файлу meta.json.
# Используем out_dir, т.к. туда сохранялись и метаданные (вместе с чекпойнтом в nanoGPT по умолчанию)
# Или можно использовать data_dir, если meta.json лежит там.
# В train.py мы сохраняли meta.json в out_dir, поэтому используем этот путь.
meta_filepath = os.path.join(out_dir, 'meta.json') # << ПУТЬ К meta.json ОТНОСИТЕЛЬНО КОРНЕВОЙ ПАПКИ NANO GPT

stoi = None
itos = None
encode = None
decode = None
bos_token_id = None
eos_token_id = None
tokenizer_model_name = None


if os.path.exists(meta_filepath):
    print(f"Загрузка метаданных из {meta_filepath}...")
    try:
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # Получаем ID специальных токенов и имя токенизатора из метаданных
        bos_token_id = meta.get('bos_token_id')
        eos_token_id = meta.get('eos_token_id')
        tokenizer_model_name = meta.get('tokenizer_model') # Имя токенизатора
        vocab_size = meta.get('vocab_size') # Размер словаря из метаданных

        if bos_token_id is None or eos_token_id is None or tokenizer_model_name is None or vocab_size is None:
            print(f"Ошибка: Файл метаданных {meta_filepath} не содержит все необходимые поля (bos_token_id, eos_token_id, tokenizer_model, vocab_size).")
            exit(1)

        print(f"Используется токенизатор: {tokenizer_model_name}")
        print(f"Размер словаря из метаданных: {vocab_size}")
        print(f"<bos> ID: {bos_token_id}, <eos> ID: {eos_token_id}")

        # Загружаем токенизатор Hugging Face
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
            # Убедимся, что специальные токены добавлены (хотя from_pretrained может сделать это сам)
            # Если у токенизатора уже есть bos_token, он не добавится заново, но его ID будет доступен
            num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': []}) # Добавляем только если они не стандартные
            if tokenizer.bos_token_id != bos_token_id or tokenizer.eos_token_id != eos_token_id:
                # На всякий случай, если ID в метаданных не совпадают со стандартными ID токенизатора
                # (редкий случай, но проверим)
                # В этом случае, возможно, нужно добавить токены с нужными ID вручную,
                # или использовать только ID из метаданных.
                # Для Gemma ID из метаданных должны совпадать со стандартными.
                print("Предупреждение: ID <bos>/<eos> из метаданных не совпадают со стандартными ID токенизатора.")
                print("Используются ID из метаданных для кодирования/декодирования.")


            # Определяем функции encode и decode с использованием загруженного токенизатора
            # add_special_tokens=False чтобы не добавлять токены BOS/EOS по умолчанию при кодировании затравки
            encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
            decode = lambda l: tokenizer.decode(l, skip_special_tokens=True) # skip_special_tokens=True чтобы <eos> не декодировался в конце

        except Exception as e:
            print(f"Ошибка при загрузке токенизатора Hugging Face '{tokenizer_model_name}': {e}")
            traceback.print_exc()
            exit(1)


    except json.JSONDecodeError:
        print(f"Ошибка: Не удалось прочитать файл метаданных как JSON: {meta_filepath}")
        exit(1)
    except Exception as e:
        print(f"Ошибка при обработке файла метаданных {meta_filepath}: {e}")
        traceback.print_exc()
        exit(1)

else:
    # Если meta.json не найден (не наш случай, но оставим для общности)
    print("Файл метаданных meta.json не найден. Невозможно загрузить токенизатор и ID специальных токенов.")
    print("Убедитесь, что prepare.py был успешно запущен и meta.json создан в директории out_dir.")
    exit(1)


# --- Кодирование начальной затравки ---
if start.startswith('FILE:'):
    try:
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл затравки не найден: {start[5:]}")
        exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла затравки: {e}")
        exit(1)


# Если start - пустая строка, начинаем с токена <bos>
if start == "":
    if bos_token_id is None:
        print("Ошибка: Невозможно начать с пустой строки, так как ID токена <bos> неизвестен.")
        exit(1)
    start_ids = [bos_token_id]
    print("Начата генерация новой статьи (с токена <bos>).")
else:
    # Иначе кодируем предоставленную строку затравки
    start_ids = encode(start)
    if not start_ids:
        print("Предупреждение: Предоставленная затравка пуста или не дала ни одного токена после кодирования.")
        # Если затравка пуста после кодирования, все равно начнем с <bos>
        if bos_token_id is not None:
            start_ids = [bos_token_id]
            print("Начата генерация новой статьи (с токена <bos>).")
        else:
            print("Ошибка: Пустая затравка после кодирования и ID токена <bos> неизвестен.")
            exit(1)
    else:
        print(f"Начата генерация с предоставленной затравки ({len(start_ids)} токенов).")


# Преобразуем список ID затравки в тензор для модели
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


# --- Запуск генерации ---
print("\nЗапуск генерации...")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # Генерация токенов моделью
            # model.generate - это метод, который мы ожидаем от модели GPT в model.py
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            # Декодирование сгенерированных ID токенов в текст
            generated_tokens = y[0].tolist() # Получаем список ID токенов из тензора

            # Находим позицию токена <eos> в сгенерированной последовательности
            if eos_token_id is not None and eos_token_id in generated_tokens:
                eos_index = generated_tokens.index(eos_token_id)
                # Обрезаем последовательность до токена <eos> (включая его)
                generated_tokens = generated_tokens[:eos_index + 1]
                print(f"Генерация остановлена по токену <eos> (после {eos_index + 1} токенов).")
            else:
                print(f"Сгенерировано максимальное количество токенов ({max_new_tokens}). Токен <eos> не найден.")

            # Декодируем обрезанную последовательность токенов в текст
            generated_text = decode(generated_tokens)

            print("\n--- СГЕНЕРИРОВАННЫЙ ТЕКСТ ---")
            print(generated_text)
            print('-----------------------------')

print("\nСэмплирование завершено.")