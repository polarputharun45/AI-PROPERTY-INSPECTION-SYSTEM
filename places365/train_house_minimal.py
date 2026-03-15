import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from wideresnet import resnet18
import os

# Minimal house classes
classes = ['bedroom', 'kitchen', 'living_room']

data_dir = 'places365/kaggle_houses'
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = resnet18(num_classes=3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Training minimal house model...")
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Epoch {epoch+1}, Val Acc: {acc:.2f}%')

torch.save(model.state_dict(), 'places365/house_minimal_trained.pth.tar')
print("Saved house_minimal_trained.pth.tar - copy to trained_from_scratch.pth.tar")

