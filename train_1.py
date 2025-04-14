import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Hàm load dữ liệu từ folder
def load_face_data(folder_path="./cropped_datasets"):
    data = []
    labels = []
    label_map = {}
    current_label = 0

    for name in sorted(os.listdir(folder_path)):
        person_path = os.path.join(folder_path, name)
        if not os.path.isdir(person_path):
            continue
        label_map[name] = current_label
        for filename in os.listdir(person_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(person_path, filename)
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (100, 100))
                data.append(resized)
                labels.append(current_label)
        current_label += 1

    return np.array(data), np.array(labels), label_map

# 2. Load và xử lý dữ liệu
data, labels, label_map = load_face_data("./cropped_datasets")
data = data.astype("float32") / 255.0  # Chuẩn hóa
data = np.expand_dims(data, axis=-1)   # Thêm kênh (100,100) -> (100,100,1)

# 3. Encode nhãn
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# 4. Tách train/test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# 5. Xây dựng mô hình CNN
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(100,100,1)))
model.add(Activation("relu"))
model.add(Conv2D(32, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(len(label_map)))
model.add(Activation("softmax"))

# 6. Compile model
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])

# # 7. Train
# print("Bắt đầu huấn luyện...")
# model.fit(X_train, y_train, batch_size=5, epochs=30, validation_data=(X_test, y_test))

# # 8. Lưu model
# model.save("model/face_recognition_model_2.h5")
# print("Huấn luyện xong và model đã lưu!")

# 9. Dữ liệu tăng cường
aug = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

import math


steps = math.ceil(len(X_train) / 5)


from sklearn.utils.class_weight import compute_class_weight

# Phục hồi nhãn gốc từ one-hot
labels_raw = np.argmax(labels, axis=1)
class_weights = compute_class_weight("balanced", classes=np.unique(labels_raw), y=labels_raw)
class_weight_dict = dict(enumerate(class_weights))

from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

model.fit(
    aug.flow(X_train, y_train, batch_size=5),
    validation_data=(X_test, y_test),
    steps_per_epoch=steps,
    epochs=50,
    callbacks=[lr_callback],
    class_weight=class_weight_dict
)

model.save("model/face_recognition_model_co_Phan.h5")
print("Huấn luyện với dữ liệu tăng cường xong và model đã lưu!")