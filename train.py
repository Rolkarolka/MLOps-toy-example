from clothPredictionModule import FashionNet
import os
import torch
from torchvision import datasets, transforms
# from torchmetrics.functional import f1, accuracy
from torchmetrics import ConfusionMatrix, Accuracy, F1Score
import torch.optim as optim
import torch.nn as nn

# training
model = FashionNet()
transform = transforms.Compose([transforms.ToTensor()])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
trainset = datasets.FashionMNIST(root="data/", train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, drop_last=True)
for i in range(10): 
    for data in trainloader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Training Epoch {i}")
print('Finished Training')
torch.save(model.state_dict(), "./saved_models/fashionNet")

# evaluation
# model.load_state_dict(torch.load("./saved_models/fashionNet"))
# model.eval()
test_dataset = datasets.FashionMNIST('./data', download=True, transform=transform, train=False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
transform = transforms.Compose([transforms.ToTensor()])
targets = []
predictions = []
with torch.no_grad():
    for i, data in enumerate(testloader):
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted)
        targets.append(labels)
print('Finished Evaulation')

targets = torch.cat(targets)
predictions = torch.cat(predictions)
# preparing raport

# accuracy_score = accuracy(predictions, targets, num_classes=10)
# print(accuracy_score)
accuracy = Accuracy(num_classes=10)
accuracy_score = accuracy(predictions, targets)

# f1_score = f1(predictions, targets, num_classes=10)
# print(f1_score)
f1 = F1Score(num_classes=10)
f1_score = f1(predictions, targets)

confmat = ConfusionMatrix(num_classes=10)
confmat(predictions, targets)  
confmat = confmat.confmat.tolist()

with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(round(accuracy_score.item(), 3)) + "\n")
    outfile.write("Confusion matrix:" + "\n")
    for l in confmat:
        outfile.write(str(l) + "\n")