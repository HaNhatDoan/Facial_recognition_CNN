import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Đường dẫn đến mô hình đã huấn luyện
MODEL_PATH = "model/face_recognition_model_1.h5"

# Hàm lấy danh sách lớp tự động từ thư mục dataset
def get_label_names(dataset_path="./datasets"):
    label_names = sorted([folder for folder in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, folder))])
    return label_names

# Tải mô hình
model = load_model(MODEL_PATH)

# Danh sách tên lớp
label_names = get_label_names("./datasets")
print("Danh sách lớp:", label_names)

# Load Haar cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tiền xử lý ảnh đầu vào
def preprocess_face(face_img):
    face = cv2.resize(face_img, (100, 100))
    face = face.astype("float32") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)  # (1, 100, 100, 1)
    return face

# Hàm nhận diện từ webcam
def predict_from_webcam():
    # Mở webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể truy cập webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]

            # Tiền xử lý khuôn mặt
            processed_face = preprocess_face(face_img)

            # Dự đoán
            prediction = model.predict(processed_face)
            class_index = np.argmax(prediction)
            class_label = label_names[class_index]
            confidence = prediction[0][class_index] * 100

            # Vẽ bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Hiển thị tên + độ tin cậy
            label_text = f"{class_label} ({confidence:.1f}%)"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Hiển thị ảnh
        cv2.imshow("Webcam - Face Recognition", frame)

        # Dừng khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Chạy nhận diện khuôn mặt từ webcam
if __name__ == "__main__":
    predict_from_webcam()
