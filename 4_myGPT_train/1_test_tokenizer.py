# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
import sys
import traceback

# --- Конфигурация ---
# Имя модели на Hugging Face, для которой загружаем токенизатор
MODEL_NAME = "Qwen/Qwen3-8B" # Используем Qwen1.5, так как Qwen3 может быть более новой или другой версией

# Qwen/Qwen3-8B может быть более новой или специфичной версией,
# если возникнут проблемы с загрузкой, попробуйте Qwen/Qwen1.5-8B-Chat
# --- Конец Конфигурации ---

# --- Шаг 1: Загрузка токенизатора ---
print("="*50)
print(f"Шаг 1: Загрузка токенизатора для модели '{MODEL_NAME}'...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Токенизатор загружен успешно.")

except Exception as e:
    print(f"\nКритическая ошибка: Не удалось загрузить токенизатор для модели '{MODEL_NAME}'.")
    print(f"Пожалуйста, убедитесь, что у вас установлена библиотека transformers (`pip install transformers`)")
    print(f"и есть доступ к интернету для загрузки токенизатора с Hugging Face.")
    print(f"Ошибка: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Шаг 2: Вывод параметров токенизатора ---
print("\n" + "="*50)
print("Шаг 2: Параметры токенизатора:")
print(f"  Размер словаря (vocab size): {tokenizer.vocab_size}")
# Можно вывести другие параметры, если интересно, например:
print(f"  Начальный токен последовательности (bos_token): {tokenizer.bos_token}")
print(f"  Конечный токен последовательности (eos_token): {tokenizer.eos_token}")
# print(f"  Токен паддинга (pad_token): {tokenizer.pad_token}")
# print(f"  Неизвестный токен (unk_token): {tokenizer.unk_token}")

print("="*50)

# --- Шаг 3: Интерактивное тестирование токенизатора ---
print("\n" + "="*50)
print("Шаг 3: Интерактивное тестирование токенизатора.")
print("Введите строку для токенизации или 'quit' для выхода.")
print("="*50)

while True:
    try:
        # Ожидаем ввод пользователя
        input_string = input("Введите текст > ")

        # Проверка на выход
        if input_string.lower() == 'quit':
            break

        if not input_string:
            print("  Введена пустая строка. Попробуйте еще.")
            print("-" * 20)
            continue

        # Токенизация строки в список ID токенов
        # add_special_tokens=False чтобы не добавлять токены начала/конца последовательности модели по умолчанию
        token_ids = tokenizer.encode(input_string, add_special_tokens=False)

        # Преобразование ID токенов обратно в их сырые строковые представления из словаря
        tokens_raw_strings = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

        # Декодирование списка ID токенов обратно в читаемую строку (для сравнения)
        decoded_full_string = tokenizer.decode(token_ids, skip_special_tokens=False)

        # --- НОВЫЙ ШАГ: Декодирование каждого токена отдельно ---
        decoded_tokens_list = []
        for token_id in token_ids:
            # Декодируем список, содержащий только один этот ID
            decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
            decoded_tokens_list.append(decoded_token)
        # --- КОНЕЦ НОВОГО ШАГА ---


        # Вывод результатов
        print("\n  Полученные ID токенов:")
        print(f"  {token_ids}")
        print("\n  Строковые представления токенов из словаря (сырые):")
        print(f"  {tokens_raw_strings}")
        print("\n  Декодированное строковое представление каждого токена:")
        print(f"  {decoded_tokens_list}") # Выводим новый список
        print("\n  Декодированная строка из ПОСЛЕДОВАТЕЛЬНОСТИ ID токенов (целиком):")
        print(f"  '{decoded_full_string}'") # Выводим полную строку в кавычках


        print("-" * 20) # Разделитель

    except Exception as e:
        print(f"\n  Произошла ошибка во время токенизации: {e}")
        traceback.print_exc()
        print("-" * 20) # Разделитель


print("\nВыход из интерактивного режима.")
print("="*50)