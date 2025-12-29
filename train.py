# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms

# ======================
# Dataset Loader
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
# CNN Model
# ======================
class SignCNN(nn.Module):
    def __init__(self, num_classes=25):
        super(SignCNN, self).__init__()
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
# Training Function
# ======================
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# ======================
# Evaluation Function
# ======================
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
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
# Main Training Script
# ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")

    print("\nEvaluating Model...")
    evaluate_model(model, test_loader, device)

    torch.save(model.state_dict(), "model/model.pth")
    print("Model saved as model.pth")


if __name__ == "__main__":
    main()
