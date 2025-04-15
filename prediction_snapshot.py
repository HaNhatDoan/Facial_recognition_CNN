import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Đường dẫn đến mô hình và thư mục dataset gốc
MODEL_PATH = "model/face_recognition_model_aug_200_epochs.h5"
DATASET_PATH = "./cropped_datasets"

# Load mô hình
model = load_model(MODEL_PATH)

# Lấy tên nhãn từ thư mục
def get_label_names(dataset_path):
    return sorted([folder for folder in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, folder))])

label_names = get_label_names(DATASET_PATH)
print("Các nhãn nhận diện:", label_names)

# Hàm tiền xử lý ảnh
def preprocess_face(face_img):
    face = cv2.resize(face_img, (100, 100))
    face = face.astype("float32") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)  # (1, 100, 100, 1)
    return face

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------
# Mở webcam và chụp ảnh
cap = cv2.VideoCapture(0)
print("Nhấn SPACE để chụp ảnh, hoặc ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Nhan dien - Bam SPACE de chup", display)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        if len(faces) == 0:
            print("Không phát hiện khuôn mặt nào!")
            continue

        # Dự đoán khuôn mặt đầu tiên
        (x, y, w, h) = faces[0]
        face_img = gray[y:y + h, x:x + w]
        processed = preprocess_face(face_img)

        # Dự đoán nhãn
        pred = model.predict(processed)[0]
        idx = np.argmax(pred)
        label = label_names[idx]
        confidence = pred[idx] * 100

        print(f"Kết quả: {label} ({confidence:.2f}%)")

        # Hiển thị ảnh đã nhận diện
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Ket qua", frame)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
