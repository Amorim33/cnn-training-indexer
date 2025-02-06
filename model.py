import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from config import config

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)  
        return x  
    
class CNNTrainingIndexer():
    def __init__(self, device):
        self.device = device
        self.model = CNN().to(device)
        self.model.load_state_dict(torch.load(f'{config['model_dir']}/{config['model_path']}', map_location=device))
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.train_features = np.load(f'{config['model_dir']}/{config['train_features_path']}')
        self.train_labels = np.load(f'{config['model_dir']}/{config['train_labels_path']}')
        self.train_images = np.load(f'{config['model_dir']}/{config['train_images_path']}')
        
    def find_most_similar_index(self, query_image):
        if isinstance(query_image, np.ndarray):
            query_image = torch.from_numpy(query_image).float()
        
        if query_image.dim() == 2:  
            query_image = query_image.unsqueeze(0)  
        
        query_image = query_image.unsqueeze(0) 
        query_image = query_image.to(self.device)

        query_feature = self.model(query_image).detach()

        similarities = F.cosine_similarity(query_feature, torch.tensor(self.train_features, dtype=torch.float32).to(self.device))

        most_similar_idx = similarities.argmax().item()

        return most_similar_idx
