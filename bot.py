import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights  # Добавляем импорт весов
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor
from PIL import Image
import os
import glob
import random
import rembg
import io
import time
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import asyncio

DF_COARSE_50 = [
    # верх 20
    "Anorak", "Blazer", "Blouse", "Bomber", "Button-Down",
    "Cardigan", "Flannel", "Halter", "Henley", "Hoodie",
    "Jacket", "Jersey", "Parka", "Peacoat", "Poncho",
    "Sweater", "Tank", "Tee", "Top", "Turtleneck",
    # низ 15
    "Capris", "Chinos", "Culottes", "Cutoffs", "Gauchos",
    "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
    "Sarong", "Shorts", "Skirt", "Sweatshorts", "Trunks",
    # full-body 15
    "Caftan", "Cape", "Coat", "Coverup", "Dress",
    "Jumpsuit", "Kaftan", "Kimono", "Nightdress", "Onesie",
    "Robe", "Romper", "Shirtdress", "Sundress", "Slacks"   # Slacks = 50-й
]

# Конфигурация
TOKEN = "7803258791:AAE1sFkqfQyQjeea-E1TImzCI4z6d9B5xuk"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация бота
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# Загрузка категорий одежды
# def load_category_names(category_file="deepfashion/Anno_coarse/list_category_cloth.txt"):
#     with open(category_file, "r") as f:
#         return [line.split()[0] for line in f.readlines()[2:]]
def load_category_names(category_file="deepfashion/Anno_coarse/list_category_cloth.txt"):
    """
    Возвращает список категорий одежды.
    • Если указанного txt-файла нет — берём имена подпапок images/*,
      а при их отсутствии — встроенный список DF_COARSE_50.
    """
#     # 1. Пытаемся прочитать файл DeepFashion
#     if os.path.exists(category_file):
#         with open(category_file, "r") as f:
#             # файл содержит заголовок из 2 строк → пропускаем
#             return [line.split()[0] for line in f.readlines()[2:]]
#
#     # 2. Файл потерян → пробуем подкаталоги images/*
#     image_root = "images"
#     if os.path.isdir(image_root):
#         subdirs = sorted(
#             d for d in os.listdir(image_root)
#             if os.path.isdir(os.path.join(image_root, d))
#         )
#         if subdirs:                             # если найдено хоть что-то
#             return subdirs
#
#     # 3. В каталоге images пусто → берём жёстко прошитый список
    return DF_COARSE_50

categories = load_category_names()


# Инициализация модели с использованием актуального API torchvision
def initialize_model():
    # Используем актуальный способ загрузки модели
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)

    # Замораживаем все слои кроме последнего
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(categories))
    model.fc.requires_grad = True

    if os.path.exists("deepfashion_resnet50.pth"):
        model.load_state_dict(torch.load("deepfashion_resnet50.pth", map_location=DEVICE))

    return model.to(DEVICE)

model = initialize_model()

# Трансформации
train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(0.1, 0.1, 0.1),
    # transforms.Resize(256),
    # transforms.RandomCrop(224),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Создаем необходимые директории
for category in categories:
    os.makedirs(f"images/{category}", exist_ok=True)
os.makedirs("temp", exist_ok=True)

def process_image(image: Image.Image) -> Image.Image:
    """Обработка изображения с удалением фона"""
    output = rembg.remove(image)
    img = Image.open(io.BytesIO(output)).convert("RGBA")
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(white_bg, img).convert("RGB")

def remove_background(image: Image.Image) -> Image.Image:
    """Удаление фона с изображения с заменой на белый"""
    try:
        # Конвертируем изображение в байты
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Удаляем фон
        output_bytes = rembg.remove(img_bytes)

        # Конвертируем обратно в изображение
        output_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

        # Создаем белый фон
        white_bg = Image.new("RGBA", output_img.size, (255, 255, 255, 255))
        result_img = Image.alpha_composite(white_bg, output_img).convert("RGB")

        return result_img
    except Exception as e:
        print(f"Error in remove_background: {str(e)}")
        return image  # Возвращаем оригинал в случае ошибки

def classify_image(image: Image.Image) -> str:
    """Классификация изображения"""
    model.eval()
    tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    return categories[pred.item()]

class BalanceDataset(Dataset):
    """Балансированный датасет для дообучения"""
    def __init__(self, new_image, category, old_ratio=0.3):
        self.new_image = new_image
        self.category_idx = categories.index(category)
        self.old_images = self._load_old_images(category, old_ratio)

    def _load_old_images(self, category, ratio):
        files = glob.glob(f"images/{category}/*.jpg")
        if not files: return []
        selected = np.random.choice(files, size=int(len(files)*ratio), replace=False)
        return [Image.open(f).convert("RGB") for f in selected]

    def __len__(self): return 64

    def __getitem__(self, idx):
        img = self.new_image if idx % 2 == 0 or not self.old_images else random.choice(self.old_images)
        return train_transform(img), torch.tensor(self.category_idx)

def safe_fine_tune(model, new_image, category):
    """Безопасное дообучение модели"""
    model.train()
    dataset = BalanceDataset(new_image, category)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(3):  # 2 эпохи
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fc.parameters(), 1.0)
            optimizer.step()

    model.eval()
    torch.save(model.state_dict(), "deepfashion_resnet50.pth")
    return model

# Хранение данных пользователей
user_data = {}
keyboard = ReplyKeyboardMarkup(resize_keyboard=True).add(
    KeyboardButton("Собрать образ"),
    KeyboardButton("Посмотреть гардероб"),
    KeyboardButton("Очистить гардероб")
)

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    """Обработчик команды /start"""
    model.load_state_dict(torch.load("deepfashion_resnet50.pth", map_location=DEVICE))
    await message.answer("Привет! Загрузи фото одежды, и я определю категорию.", reply_markup=keyboard)

# Добавляем в начало кода (после user_data)
user_stacks = {}  # Словарь для хранения стеков пользователей

def get_user_stack(user_id):
    """Получаем или создаем стек для пользователя"""
    if user_id not in user_stacks:
        user_stacks[user_id] = []
    return user_stacks[user_id]

# Модифицируем обработчик фото
@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    """Обработка загруженного фото"""
    try:
        user_id = message.from_user.id
        file_id = message.photo[-1].file_id

        # Добавляем file_id в стек пользователя
        user_stack = get_user_stack(user_id)
        user_stack.append(file_id)

        # Скачиваем фото
        file = await message.photo[-1].get_file()
        img_path = f"temp/{file_id}.jpg"
        await file.download(img_path)

        # Открываем и обрабатываем изображение
        image = Image.open(img_path).convert("RGB")
        processed_image = remove_background(image)

        # Классифицируем изображение
        category = classify_image(processed_image)

        # Сохраняем данные для последующего подтверждения
        user_data[file_id] = {
            'image': processed_image,
            'category': category,
            'path': img_path,
            'user_id': user_id
        }

        # Запрашиваем подтверждение
        markup = ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add("Да", "Нет")
        await message.answer(f"Это {category}?", reply_markup=markup)

    except Exception as e:
        await message.answer(f"Ошибка обработки фото: {str(e)}")
        print(f"Error in handle_photo: {str(e)}")

# Модифицируем обработчики подтверждения
@dp.message_handler(lambda m: m.text in ["Да", "Нет"])
async def handle_confirmation(message: types.Message):
    user_id = message.from_user.id
    user_stack = get_user_stack(user_id)

    if message.text == "Да":
        if not user_stack:
            return await message.answer("Нет активных фото для обработки!")

        file_id = user_stack.pop()  # Берем последнее фото из стека
        data = user_data.get(file_id)

        if not data:
            return await message.answer("Данные фото не найдены!")

        category = data['category']
        save_path = f"images/{category}/{int(time.time())}.jpg"
        data['image'].save(save_path)
        # global model
        # model = safe_fine_tune(model, data['image'], category)
        await message.answer("Модель обновлена! ✅", reply_markup=keyboard)
    else:
        markup = ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(*categories)
        await message.answer("Выберите категорию:", reply_markup=markup)

    # Не удаляем данные сразу, они могут понадобиться для выбора категории
    # Очистка будет после окончательного сохранения

@dp.message_handler(lambda message: message.text in categories)
async def handle_category_selection(message: types.Message):
    user_id = message.from_user.id
    user_stack = get_user_stack(user_id)

    if not user_stack:
        return await message.answer("Нет активных фото для обработки!")

    file_id = user_stack[-1]  # Берем текущее фото (не удаляем из стека)
    data = user_data.get(file_id)

    if not data:
        return await message.answer("Данные фото не найдены!")

    category = message.text
    save_path = f"images/{category}/{int(time.time())}.jpg"
    data['image'].save(save_path)

    global model
    model = safe_fine_tune(model, data['image'], category)

    # Удаляем фото из стека и данных после сохранения
    user_stack.pop()
    del user_data[file_id]

    await message.answer(f"Фото сохранено в {category}! ✅", reply_markup=keyboard)


@dp.message_handler(lambda message: message.text == "Собрать образ")
async def assemble_outfit(message: types.Message):
    try:
        # Классификация всех 50 классов DeepFashion
        WARM_TOPS = ["Anorak", "Bomber", "Cardigan", "Flannel", "Hoodie",
                    "Jacket", "Parka", "Peacoat", "Poncho", "Sweater", "Turtleneck"]
        FORMAL_TOPS = ["Blazer", "Blouse", "Button-Down", "Shirtdress"]
        SPORT_TOPS = ["Jersey", "Tank", "Halter", "Top"]
        CASUAL_TOPS = ["Tee", "Henley", "Polo"]
        FULL_BODY = ["Dress", "Jumpsuit", "Romper", "Caftan", "Cape",
                    "Coverup", "Kaftan", "Kimono", "Nightdress", "Onesie",
                    "Robe", "Sundress"]

        SPORT_BOTTOMS = ["Joggers", "Leggings", "Sweatpants", "Sweatshorts", "Trunks"]
        FORMAL_BOTTOMS = ["Chinos", "Slacks"]
        CASUAL_BOTTOMS = ["Jeans", "Jeggings", "Capris", "Culottes", "Cutoffs",
                         "Gauchos", "Jodhpurs", "Sarong", "Shorts", "Skirt"]

        # Собираем файлы по категориям
        def gather_files(categories):
            files = []
            for cat in categories:
                cat_files = glob.glob(f"images/{cat}/*.jpg") + glob.glob(f"images/{cat}/*.png")
                files.extend(cat_files)
            return files

        # Получаем все доступные вещи
        warm_tops = gather_files(WARM_TOPS)
        formal_tops = gather_files(FORMAL_TOPS)
        sport_tops = gather_files(SPORT_TOPS)
        casual_tops = gather_files(CASUAL_TOPS)
        full_body = gather_files(FULL_BODY)

        sport_bottoms = gather_files(SPORT_BOTTOMS)
        formal_bottoms = gather_files(FORMAL_BOTTOMS)
        casual_bottoms = gather_files(CASUAL_BOTTOMS)

        # Сначала проверяем full body - они не требуют пары
        if full_body and random.random() < 0.3:  # 30% chance выбрать full body
            selected_item = random.choice(full_body)
            collage = Image.new('RGB', (256, 512))
            img = Image.open(selected_item).resize((256, 512))
            collage.paste(img, (0, 0))
        else:
            # Собираем все возможные верхи с их категориями
            top_options = []
            if warm_tops: top_options.extend([("warm", top) for top in warm_tops])
            if formal_tops: top_options.extend([("formal", top) for top in formal_tops])
            if sport_tops: top_options.extend([("sport", top) for top in sport_tops])
            if casual_tops: top_options.extend([("casual", top) for top in casual_tops])

            if not top_options:
                return await message.answer("Нет доступных верхних вещей 😢")

            # Перемешиваем варианты верхов для случайного выбора
            random.shuffle(top_options)

            # Флаг, что образ собран
            outfit_assembled = False

            for top_category, selected_top in top_options:
                # Пытаемся подобрать низ для текущего верха
                bottom_options = []

                # 1. Теплый верх (не сочетается с шортами и спортивными низами)
                if top_category == "warm":
                    bottom_options.extend(formal_bottoms)
                    bottom_options.extend([b for b in casual_bottoms
                                        if not any(x in b for x in ["Shorts", "Cutoffs", "Trunks"])])

                # 2. Формальный верх (только формальные/классические низы)
                elif top_category == "formal":
                    bottom_options.extend(formal_bottoms)
                    bottom_options.extend([b for b in casual_bottoms
                                         if not any(x in b for x in ["Shorts", "Sweatshorts", "Cutoffs"])])

                # 3. Спортивный верх (только спортивные низы и некоторые casual)
                elif top_category == "sport":
                    bottom_options.extend(sport_bottoms)
                    bottom_options.extend([b for b in casual_bottoms
                                         if any(x in b for x in ["Shorts", "Sweatshorts", "Joggers"])])

                # 4. Casual верх (почти любые низы, кроме явно несочетаемых)
                elif top_category == "casual":
                    bottom_options.extend(casual_bottoms)
                    if "Tee" not in selected_top:  # Если не простая футболка
                        bottom_options.extend(formal_bottoms)
                    bottom_options.extend([b for b in sport_bottoms
                                         if any(x in b for x in ["Shorts", "Sweatshorts"])])

                # Удаляем дубликаты
                bottom_options = list(set(bottom_options))

                if bottom_options:
                    # Нашли подходящие низы - собираем образ
                    selected_bottom = random.choice(bottom_options)
                    outfit_assembled = True
                    break

            if not outfit_assembled:
                return await message.answer("Не удалось собрать образ из имеющихся вещей 😢")

            # Создаем коллаж (верх + низ)
            collage = Image.new('RGB', (256, 512))
            collage.paste(Image.open(selected_top).resize((256, 256)), (0, 0))
            collage.paste(Image.open(selected_bottom).resize((256, 256)), (0, 256))

        # Сохраняем и отправляем результат
        timestamp = int(time.time())
        collage_path = f"temp/outfit_{timestamp}.jpg"
        os.makedirs("temp", exist_ok=True)
        collage.save(collage_path)

        with open(collage_path, 'rb') as photo:
            await message.answer_photo(photo, caption="Ваш стильный образ!")

        os.remove(collage_path)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        await message.answer("Ошибка: некоторые файлы одежды не найдены")
    except PIL.UnidentifiedImageError:
        await message.answer("Ошибка обработки изображений")
    except Exception as e:
        logger.error(f"Ошибка при создании образа: {e}")
        await message.answer("Произошла ошибка при создании образа 😢")

@dp.message_handler(lambda message: message.text == "Посмотреть гардероб")
async def view_closet(message: types.Message):
    try:
        for category in categories:
            files = glob.glob(f"images/{category}/*.jpg")
            if not files: continue

            await message.answer(f"📁 {category}:")
            for file in files[:3]:
                with open(file, 'rb') as photo:
                    await message.answer_photo(photo)
                    await asyncio.sleep(0.3)
    except Exception as e:
        await message.answer("Ошибка просмотра 😢")

@dp.message_handler(lambda message: message.text == "Очистить гардероб")
async def clear_closet(message: types.Message):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("Подтвердить очистку", "Отмена")
    await message.answer("Вы уверены? Это действие необратимо!", reply_markup=markup)

@dp.message_handler(lambda message: message.text == "Подтвердить очистку")
async def confirm_clear(message: types.Message):
    count = 0
    for category in categories:
        files = glob.glob(f"images/{category}/*.jpg")
        for file in files:
            os.remove(file)
            count += 1
    await message.answer(f"Удалено {count} предметов! 🗑️", reply_markup=keyboard)

@dp.message_handler(lambda message: message.text == "Отмена")
async def cancel_clear(message: types.Message):
    await message.answer("Очистка отменена ✅", reply_markup=keyboard)

async def on_startup(_):
    """Действия при запуске бота"""
    await bot.set_my_commands([
        types.BotCommand(command="/start", description="Запустить бота")
    ])

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)
