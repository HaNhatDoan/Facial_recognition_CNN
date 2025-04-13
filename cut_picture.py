import cv2
import os

# Thư mục chứa ảnh gốc
INPUT_DIR = "./datasets/Doan"
# Thư mục chứa ảnh khuôn mặt sau khi cắt
OUTPUT_DIR = "./cropped_datasets/Doan"

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Haar cascade detector cho khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Duyệt qua các file ảnh trong thư mục input
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"⚠️ Không phát hiện khuôn mặt trong {filename}")
            continue

        # Cắt từng khuôn mặt và lưu
        for i, (x, y, w, h) in enumerate(faces):
            face_img = img[y:y+h, x:x+w]
            output_filename = f"{os.path.splitext(filename)[0]}_face{i+1}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, face_img)
            print(f"✅ Đã lưu khuôn mặt: {output_filename}")

print("🎉 Xong!")
