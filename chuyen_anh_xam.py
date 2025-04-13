from PIL import Image
import os

def convert_images_to_gray(input_folder, output_folder):
    # Kiểm tra nếu thư mục đầu ra không tồn tại, tạo thư mục mới
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Duyệt qua tất cả các tệp trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Kiểm tra nếu tệp là hình ảnh
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Mở ảnh và chuyển sang ảnh xám
            img = Image.open(file_path).convert('L')
            
            # Lưu ảnh xám vào thư mục đầu ra
            output_path = os.path.join(output_folder, filename)
            img.save(output_path)
            print(f"Đã chuyển đổi: {filename}")

# Đường dẫn đến thư mục chứa ảnh đầu vào và thư mục lưu ảnh xám
input_folder = 'D:/Dowload/Picture/Tai/Tai'
output_folder = 'D:/GIT/Facial_recognition_CNN/datasets/Tai'

# Gọi hàm chuyển đổi ảnh
convert_images_to_gray(input_folder, output_folder)
