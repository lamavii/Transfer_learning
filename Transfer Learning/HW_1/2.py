#2) Скачать нейросеть BERT для лингвистических задач и реализовать процедуру классификации текстов

#В терминале пишем pip install transformers torch

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Загрузка предобученной модели BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()  # Установка режима оценки


def classify_text(text):
    # Токенизация входного текста
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Выполнение предсказания
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


# Пример использования
text_sample = "Ваш текст для классификации."
predicted_class = classify_text(text_sample)
print(f'Предсказанный класс текста: {predicted_class}')