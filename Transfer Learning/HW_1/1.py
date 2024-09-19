#1) Скачать нейросеть ResNet и написать процедуру для предсказания класса изображения

# В терминале пишем pip install torch torchvision

import torch
from torchvision import models, transforms
from PIL import Image

# Загрузка предобученной модели ResNet
model = models.resnet50(pretrained=True)
model.eval()  # Установка режима оценки

# Определение преобразований для изображений
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image_class(image_path):
    # Загрузка и подготовка изображения
    img = Image.open(image_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # Выполнение предсказания
    with torch.no_grad():
        out = model(batch_t)

    # Получение предсказанного класса
    _, predicted = torch.max(out, 1)
    return predicted.item()


# Пример использования
image_path = 'path/to/your/image.jpg'
predicted_class = predict_image_class(image_path)
print(f'Предсказанный класс: {predicted_class}')

