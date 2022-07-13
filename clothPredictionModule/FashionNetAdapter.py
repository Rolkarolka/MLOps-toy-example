
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.optim as optim

import torch
import torch.nn as nn

import os

from clothPredictionModule import FashionNet

class FashionNetAdapter:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])
        batch_size = 256
        self.dir_path = "./saved_models"
        self.model_path = self.dir_path + "/fashionNet"
        self.trainset = FashionMNIST(root="data/", train=True, transform=self.transform, download=True)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.fashion_net = self._prepare_model()


    def _prepare_model(self):
        model = FashionNet()
        if os.path.isfile(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
        else:
            model = self._train_model(model)
            os.mkdir(self.dir_path)
            torch.save(model.state_dict(), self.model_path)
        return model

    def predict(self, net_input):
        self.fashion_net.eval()
        with torch.no_grad():
            images, _ = net_input
            images = self.transform(images).unsqueeze(0)
            outputs = self.fashion_net(images)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.item()

    def _train_model(self, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        for _ in range(2): 
            for data in self.trainloader:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        print('Finished Training')
        return model

