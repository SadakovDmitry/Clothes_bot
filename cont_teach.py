import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from tqdm import tqdm

# Гиперпараметры
BATCH_SIZE = 32
EPOCHS = 5  # Общее количество эпох (например, 5, если ты хочешь обучать 5 эпох)
LR = 0.0001
NUM_CLASSES = 50  # Количество классов в DeepFashion
DATASET_PATH = "deepfashion"
IMG_DIR = os.path.join(DATASET_PATH, "img")
CATEGORY_FILE = os.path.join(DATASET_PATH, "Anno_coarse", "list_category_img.txt")
CATEGORY_CLOTH_FILE = os.path.join(DATASET_PATH, "Anno_coarse", "list_category_cloth.txt")
PARTITION_FILE = os.path.join(DATASET_PATH, "Eval", "list_eval_partition.txt")
BBOX_FILE = os.path.join(DATASET_PATH, "Anno_coarse", "list_bbox.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Трансформации изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Читаем категории
def read_category_file(category_file):
    with open(category_file, "r") as f:
        lines = f.readlines()[2:]  # Пропускаем заголовки
        category_map = {line.split()[0]: int(line.split()[1]) - 1 for line in lines}
    return category_map

category_map = read_category_file(CATEGORY_FILE)
category_cloth_map = read_category_file(CATEGORY_CLOTH_FILE)

# Читаем разбиение на train/val/test
def read_partition_file(partition_file):
    with open(partition_file, "r") as f:
        lines = f.readlines()[2:]  # Пропускаем заголовки
        partitions = {line.split()[0]: line.split()[1] for line in lines}
    return partitions

partitions = read_partition_file(PARTITION_FILE)

# Читаем bounding box
def read_bbox_file(bbox_file):
    with open(bbox_file, "r") as f:
        lines = f.readlines()[2:]  # Пропускаем заголовки
        bbox_map = {}
        for line in lines:
            parts = line.split()
            bbox_map[parts[0]] = list(map(int, parts[1:]))  # (x1, y1, x2, y2)
    return bbox_map

bbox_map = read_bbox_file(BBOX_FILE)

# Кастомный датасет
class DeepFashionDataset(Dataset):
    def __init__(self, img_dir, category_map, partitions, bbox_map, partition, transform=None):
        self.img_dir = img_dir
        self.category_map = category_map
        self.bbox_map = bbox_map
        self.partition = partition
        self.transform = transform
        self.images = [img for img, part in partitions.items() if part == partition]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img_path = os.path.normpath(img_path)  # Исправляем возможные ошибки с путями

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Изображение не найдено: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # Обрезаем изображение по bounding box
        if img_name in self.bbox_map:
            x1, y1, x2, y2 = self.bbox_map[img_name]
            image = image.crop((x1, y1, x2, y2))

        # Получаем метку категории для изображения
        label = self.category_map[img_name]

        if self.transform:
            image = self.transform(image)
        return image, label

# Создаём датасеты и загрузчики
datasets = {
    part: DeepFashionDataset(IMG_DIR, category_map, partitions, bbox_map, part, transform)
    for part in ["train", "val", "test"]
}

dataloaders = {
    part: DataLoader(datasets[part], batch_size=BATCH_SIZE, shuffle=(part == "train"))
    for part in ["train", "val", "test"]
}

# Модель ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Замораживаем все слои до последнего
# for param in model.parameters():
#     param.requires_grad = False

# Размораживаем только последний слой (fc)
# model.fc.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # Подгоняем под количество классов в DeepFashion
model.to(device)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Загрузка сохранённой модели и оптимизатора
def load_checkpoint():
    if os.path.exists("deepfashion_resnet50.pth"):
        model.load_state_dict(torch.load("deepfashion_resnet50.pth"))
        print("Модель успешно загружена.")
    if os.path.exists("optimizer.pth"):
        optimizer.load_state_dict(torch.load("optimizer.pth"))
        print("Состояние оптимизатора загружено.")

load_checkpoint()

# Функция для сохранения модели и оптимизатора
def save_checkpoint(epoch):
    torch.save(model.state_dict(), "deepfashion_resnet50.pth")
    torch.save(optimizer.state_dict(), "optimizer.pth")
    print(f"Модель и оптимизатор сохранены на эпохе {epoch+1}!")

# Обучение
starting_epoch = 2  # Начать с этой эпохи, если модель уже была загружена
for epoch in range(starting_epoch, EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    progress_bar = tqdm(dataloaders["train"], desc=f"Эпоха {epoch+1}/{EPOCHS}", leave=True)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=total_loss / (total / BATCH_SIZE), accuracy=100 * correct / total)

    print(f"Эпоха {epoch+1}/{EPOCHS}, Потери: {total_loss/len(dataloaders['train']):.4f}, "
          f"Точность: {100 * correct / total:.2f}%")

    # Сохранение после каждой эпохи
    save_checkpoint(epoch)
