import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#(Sửa đường dẫn)

# Kiểm tra load cascade thành công không
if face.empty():
    print("[ERROR] Không load được haarcascade")
    exit()

# Kiểm tra mở camera thành công không
if not cam.isOpened():
    print("[ERROR] Không mở được camera")
    exit()
    
face_id = input("/n Nhap ID khuon mat <return> ==>")
print ("/n [INFO] Khoi tao camera")
count = 0

while (True):
    
    ret, img = cam.read()
    
    if not ret:
        print("[ERROR] Không đọc được frame")
        break
    
    img = cv2.flip(img , 1)
    gray = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite("datasets/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        #(Sửa đuờng dẫn)
    cv2.imshow('image', img)
    
    k = cv2.waitKey(100) & 0xff
    
    if k == 27:
        break
    elif count >= 100:
        break

print("/n [INFO] Ket thuc khoi tao khuon mat")
cam.release()
cv2.destroyAllWindows()