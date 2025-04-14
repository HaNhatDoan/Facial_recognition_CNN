import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Đường dẫn model
MODEL_PATH = "model/face_recognition_model_co_Phan.h5"

# Hàm lấy nhãn từ folder
def get_label_names(dataset_path="./cropped_datasets"):
    label_names = sorted([folder for folder in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, folder))])
    return label_names

# Load model và nhãn
model = load_model(MODEL_PATH)
label_names = get_label_names("./cropped_datasets")
print("Danh sách lớp:", label_names)

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tiền xử lý ảnh (grayscale chuẩn input model)
def preprocess_face(face_img):
    face = cv2.resize(face_img, (100, 100))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)       # (100, 100, 1)
    face = np.expand_dims(face, axis=0)        # (1, 100, 100, 1)
    return face

# Webcam nhận diện
def predict_from_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            processed_face = preprocess_face(face_img)

            prediction = model.predict(processed_face, verbose=0)
            class_index = np.argmax(prediction)
            class_label = label_names[class_index]
            confidence = prediction[0][class_index] * 100

            # Vẽ bounding box và label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label_text = f"{class_label} ({confidence:.1f}%)"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Webcam - Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy nhận diện
if __name__ == "__main__":
    predict_from_webcam()
