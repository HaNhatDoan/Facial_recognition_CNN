import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modelCNN import FaceCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(),                    # Ảnh xám
    transforms.Resize((128, 128)),             # Resize
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

data_dir = "datasets"
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
num_classes = len(dataset.classes)

model = FaceCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    total_loss = 0
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/10] - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "face_cnn_gray.pth")
print("✅ Mô hình đã lưu thành face_cnn_gray.pth")
