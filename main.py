import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import warnings
import pickle

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

train_dir = r"E:\Plant_Identifier_by_Leaves\Plants_2\train"
test_dir = r"E:\Plant_Identifier_by_Leaves\Plants_2\test"

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Train directory '{train_dir}' not found.")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory '{test_dir}' not found.")

img_height, img_width = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

class_names = list(train_data.class_indices.keys())
print(f"Plant classes: {class_names}")

base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 1
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs
)

model_path = r"E:\Plant_Identifier_by_Leaves\plant_leaf_classifier.h5"
model.save(model_path)
print(f"Model saved as {model_path}")

class_names_path = r"E:\Plant_Identifier_by_Leaves\class_names.pkl"
with open(class_names_path, 'wb') as f:
    pickle.dump(class_names, f)
print(f"Class names list saved: {class_names_path}")
