import os
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from model import CNN
from config import config


os.makedirs(config['model_dir'], exist_ok=True)

def main():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(10): 
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            features = model(images)
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()

    model_save_path = os.path.join(config['model_dir'], config['model_path'])
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    model.eval()
    train_features = []
    train_labels = []
    train_images = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            features = model(images).cpu().numpy()
            train_features.append(features)
            train_labels.append(labels.numpy())
            train_images.append(images.squeeze(1).cpu().numpy()) 

    train_features = np.vstack(train_features)
    train_labels = np.hstack(train_labels)
    train_images = np.vstack(train_images)

    features_save_path = os.path.join(config['model_dir'], config['train_features_path'])
    labels_save_path = os.path.join(config['model_dir'], config['train_labels_path'])
    images_save_path = os.path.join(config['model_dir'], config['train_images_path'])
    np.save(features_save_path, train_features)
    np.save(labels_save_path, train_labels)
    np.save(images_save_path, train_images)
    print(f"Training features saved to {features_save_path}")
    print(f"Training labels saved to {labels_save_path}")
    print(f"Training images saved to {images_save_path}")


if __name__ == "__main__":
    main()
