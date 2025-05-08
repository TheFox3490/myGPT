import datasets
import os

# Название датасета на Hugging Face
dataset_name = "google/wiki40b"
# Языковое подмножество, которое нам нужно (русский)
language_subset = "ru"
# Название локальной папки для сохранения данных
# Создастся папка с именем, например, "google_wiki40b_ru"
save_directory = f"./{dataset_name.replace('/', '_')}_{language_subset}"

print(f"Загрузка датасета '{dataset_name}' подмножество '{language_subset}'...")

try:
    # Загружаем указанное подмножество (русский язык) и все его разделы (train, validation, test)
    # Это вернет DatasetDict, где ключи - названия разделов
    wiki_dataset = datasets.load_dataset(dataset_name, language_subset)

    # Создаем локальную папку, если она еще не существует
    os.makedirs(save_directory, exist_ok=True)
    print(f"Данные будут сохранены в папку: {save_directory}")

    # Сохраняем каждый раздел датасета в отдельную подпапку
    for split_name, dataset_split in wiki_dataset.items():
        split_save_path = os.path.join(save_directory, split_name)
        print(f"Сохранение раздела '{split_name}' в '{split_save_path}'...")
        dataset_split.save_to_disk(split_save_path)
        print(f"Раздел '{split_name}' успешно сохранен.")

    print("Загрузка и сохранение завершены.")

except Exception as e:
    print(f"Произошла ошибка: {e}")
    print("Пожалуйста, убедитесь, что у вас установлено 'datasets' (`pip install datasets`)")
    print("и что у вас есть активное подключение к интернету.")