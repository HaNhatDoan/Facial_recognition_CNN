import torch
from torchvision import transforms
from PIL import Image
from modelCNN import FaceCNN
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageFolder("datasets")
class_names = dataset.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FaceCNN(len(class_names))
model.load_state_dict(torch.load("face_cnn_gray.pth", map_location=device))
model.to(device)
model.eval()

def predict(img_path):
    img = Image.open(img_path).convert("L")  # Grayscale
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    print(f"Dự đoán: {class_names[pred.item()]}")

if __name__ == "__main__":
    path = input("Nhập đường dẫn ảnh cần dự đoán: ")
    predict(path)
