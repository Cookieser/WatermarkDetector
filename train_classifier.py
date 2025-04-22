import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm


os.makedirs("res", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)  
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  

model = model.to(device)

class WatermarkDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        original_dir = os.path.join(root, "original")
        watermarked_dir = os.path.join(root, "watermarked")

        # label: 0 for original, 1 for watermarked
        for fname in os.listdir(original_dir):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                self.image_paths.append(os.path.join(original_dir, fname))
                self.labels.append(0)

        for fname in os.listdir(watermarked_dir):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                self.image_paths.append(os.path.join(watermarked_dir, fname))
                self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_dataset = WatermarkDataset(root="/work/forgery/Data/StableSignature", transform=transform)

# 80% train, 20% val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

EPOCHS = 100
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    val_loss, val_acc = evaluate(model, val_loader, criterion)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    torch.save(model, f'res/resnet18_watermark_classifier_full_{epoch}.pth')

    print(f"[{epoch+1}/{EPOCHS}] Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")


import matplotlib.pyplot as plt

def plot_training_history(train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label='Train Accuracy')
    plt.plot(epochs, val_acc_list, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("res/training_plot.png")
    # plt.show()

plot_training_history(train_loss_list, val_loss_list, train_acc_list, val_acc_list)