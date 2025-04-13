import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Đường dẫn model và ảnh
MODEL_PATH = "model/face_recognition_model_2.h5"
IMAGE_PATH = "D:/GIT/Facial_recognition_CNN/datasets/Tai/IMG_9600.JPEG"

# Tải mô hình
model = load_model(MODEL_PATH)

# Tự động lấy nhãn từ tên folder trong dataset
def get_label_names(dataset_path="./datasets"):
    label_names = sorted([folder for folder in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, folder))])
    return label_names

label_names = get_label_names("./datasets")

# Hàm tiền xử lý khuôn mặt
def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    norm_img = resized.astype("float32") / 255.0
    img_array = img_to_array(norm_img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 100, 100, 1)
    return img_array

# Tải ảnh
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("❌ Không tìm thấy ảnh.")
    exit()

# Tải bộ phát hiện khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Chuyển ảnh sang xám để phát hiện khuôn mặt
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Nhận diện từng khuôn mặt
for (x, y, w, h) in faces:
    face_img = image[y:y+h, x:x+w]  # Cắt vùng khuôn mặt
    processed = preprocess_face(face_img)
    prediction = model.predict(processed)
    class_index = np.argmax(prediction)
    label = label_names[class_index]
    confidence = prediction[0][class_index] * 100

    # Vẽ bounding box và nhãn
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, f"{label} ({confidence:.2f}%)", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Resize ảnh để hiển thị nhỏ hơn (ví dụ: 60% kích thước gốc)
scale_percent = 40  # giảm còn 60%
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Dự đoán
prediction = model.predict(processed)
class_index = np.argmax(prediction)
class_label = label_names[class_index]
confidence = prediction[0][class_index] * 100

# ✅ In ra console
print(f"📢 Dự đoán: {class_label} ({confidence:.2f}%)")

cv2.imshow("Result - Face Recognition", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()