import os
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

img_height, img_width = 224, 224

model_path = r"E:\Plant_Identifier_by_Leaves\plant_leaf_classifier.h5"
class_names_path = r"E:\Plant_Identifier_by_Leaves\class_names.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Модель по пути '{model_path}' не найдена.")
if not os.path.exists(class_names_path):
    raise FileNotFoundError(f"Список классов по пути '{class_names_path}' не найден.")

model = load_model(model_path)
with open(class_names_path, 'rb') as f:
    class_names = pickle.load(f)
print("Модель и классы успешно загружены!")

image_path = r"E:\Plant_Identifier_by_Leaves\Plants_2\images to predict\0001_0170.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Изображение по пути '{image_path}' не найдено.")

image = load_img(image_path, target_size=(img_height, img_width))
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = image_array / 255.0

predictions = model.predict(image_array)
predicted_class_idx = np.argmax(predictions, axis=1)[0]
predicted_class_name = class_names[predicted_class_idx]

print(f"Вероятности предсказания: {predictions[0]}")
print(f"Модель предсказала, что это: {predicted_class_name}")
