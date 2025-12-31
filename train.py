import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms


# ======================
# DATASET
# ======================
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        self.labels = data.iloc[:, 0].values
        self.images = data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


# ======================
# CNN MODEL
# ======================
class SignCNN(nn.Module):
    def __init__(self, num_classes=25):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================
# TRAINING
# ======================
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ======================
# EVALUATION
# ======================
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return acc


# ======================
# MAIN
# ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = SignLanguageDataset("dataset/sign_mnist_train.csv")
    test_dataset = SignLanguageDataset("dataset/sign_mnist_test.csv")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SignCNN(num_classes=25).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss:.4f}")

    print("\nEvaluating model...")
    evaluate_model(model, test_loader, device)

    torch.save(model.state_dict(), "model/model.pth")
    print("✓ Model saved to model/model.pth")


if __name__ == "__main__":
    main()

# #2
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
#
# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score, classification_report
# from torchvision import transforms
#
#
# # ======================
# # DATASET
# # ======================
# class SignLanguageDataset(Dataset):
#     def __init__(self, csv_file, augment=False):
#         data = pd.read_csv(csv_file)
#
#         self.labels = data.iloc[:, 0].values
#         self.images = data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
#         self.augment = augment
#
#         # Transform dasar
#         self.base_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#
#         # Transform dengan augmentasi untuk training
#         self.augment_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomRotation(15),
#             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image = self.images[idx]
#
#         if self.augment:
#             image = self.augment_transform(image)
#         else:
#             image = self.base_transform(image)
#
#         label = self.labels[idx]
#         return image, label
#
#
# # ======================
# # CNN MODEL (OPTIMIZED)
# # ======================
# class SignCNN(nn.Module):
#     def __init__(self, num_classes=25):
#         super().__init__()
#
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1),
#             nn.BatchNorm2d(32),  # Tambahan
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),  # Tambahan
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             # Layer tambahan
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 3 * 3, 512),  # Updated size
#             nn.BatchNorm1d(512),  # Tambahan
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),  # Layer tambahan
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
#
#
# # ======================
# # TRAINING
# # ======================
# def train_model(model, loader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#
#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#         # Tambahan: hitung akurasi training
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     avg_loss = total_loss / len(loader)
#     accuracy = 100 * correct / total
#     return avg_loss, accuracy
#
#
# # ======================
# # EVALUATION
# # ======================
# def evaluate_model(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []
#
#     with torch.no_grad():
#         for images, labels in loader:
#             images = images.to(device)
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
#
#             y_true.extend(labels.numpy())
#             y_pred.extend(preds.cpu().numpy())
#
#     acc = accuracy_score(y_true, y_pred)
#     print("Accuracy:", acc)
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred))
#
#     return acc
#
#
# # ======================
# # MAIN
# # ======================
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)
#
#     # Dataset dengan augmentasi untuk training
#     train_dataset = SignLanguageDataset("dataset/sign_mnist_train.csv", augment=True)
#     test_dataset = SignLanguageDataset("dataset/sign_mnist_test.csv", augment=False)
#
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
#
#     model = SignCNN(num_classes=25).to(device)
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
#
#     epochs = 15
#     best_acc = 0.0
#
#     for epoch in range(epochs):
#         # Simpan learning rate sebelum update
#         current_lr = optimizer.param_groups[0]['lr']
#
#         loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
#         print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss:.4f} - Train Acc: {train_acc:.2f}% - LR: {current_lr:.6f}")
#
#         # Update learning rate
#         scheduler.step(loss)
#
#         # Check jika learning rate berubah
#         new_lr = optimizer.param_groups[0]['lr']
#         if new_lr != current_lr:
#             print(f"Learning rate reduced to {new_lr:.6f}")
#
#         # Evaluasi setiap 3 epoch
#         if (epoch + 1) % 3 == 0:
#             print(f"\nValidation at epoch {epoch + 1}:")
#             val_acc = evaluate_model(model, test_loader, device)
#
#             # Simpan model terbaik
#             if val_acc > best_acc:
#                 best_acc = val_acc
#                 torch.save(model.state_dict(), "model/model_best.pth")
#                 print(f"✓ Best model saved with accuracy: {best_acc:.4f}\n")
#
#     print("\nFinal Evaluation:")
#     evaluate_model(model, test_loader, device)
#
#     torch.save(model.state_dict(), "model/model.pth")
#     print("✓ Final model saved to model/model.pth")
#
#
# if __name__ == "__main__":
#     main()