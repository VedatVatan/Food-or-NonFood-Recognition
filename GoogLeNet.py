import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import googlenet
from PIL import Image

#Perform data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224 dimensions
    transforms.ToTensor(),          # Convert the images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization
])


# #Create a custom dataset. We use the Food_5K Dataset
class CustomDataset(Dataset):
    def __init__(self, root, label, transform=None):
        self.root = root
        self.label = label
        self.transform = transform
        self.images = os.listdir(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label


# Specifying the paths of the folders containing images
food_train_data_path = r'C:\Users\vedat\Food_5k\Food\Training'
nonfood_train_data_path = r'C:\Users\vedat\Food_5k\Nonfood\Training'
food_test_data_path = r'C:\Users\vedat\Food_5k\Food\Test'
nonfood_test_data_path = r'C:\Users\vedat\Food_5k\Nonfood\Test'
food_val_data_path = r'C:\Users\vedat\Food_5k\Food\Validation'
nonfood_val_data_path = r'C:\Users\vedat\Food_5k\Nonfood\Validation'

# Load the custom training and validation datasets.
food_train_dataset = CustomDataset(root=food_train_data_path, label=0, transform=transform)
nonfood_train_dataset = CustomDataset(root=nonfood_train_data_path, label=1, transform=transform)
food_test_dataset = CustomDataset(root=food_test_data_path, label=0, transform=transform)
nonfood_test_dataset = CustomDataset(root=nonfood_test_data_path, label=1, transform=transform)
food_val_dataset = CustomDataset(root=food_val_data_path, label=0, transform=transform)
nonfood_val_dataset = CustomDataset(root=nonfood_val_data_path, label=1, transform=transform)

# Create DataLoaders
batch_size = 32
food_train_loader = DataLoader(food_train_dataset, batch_size=batch_size, shuffle=True)
nonfood_train_loader = DataLoader(nonfood_train_dataset, batch_size=batch_size, shuffle=True)
food_test_loader = DataLoader(food_test_dataset, batch_size=batch_size, shuffle=False)
nonfood_test_loader = DataLoader(nonfood_test_dataset, batch_size=batch_size, shuffle=False)
food_val_loader = DataLoader(food_val_dataset, batch_size=batch_size, shuffle=False)
nonfood_val_loader = DataLoader(nonfood_val_dataset, batch_size=batch_size, shuffle=False)

# Specify class labels
classes = ['Food', 'NonFood']

# Load the pretrained GoogleNet model
model = googlenet(pretrained=True)
# Change the class output layer of the model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))

# Specify the optimization process and loss function for training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for food_inputs, _ in food_train_loader:
        food_inputs = food_inputs.to(device)
        food_labels = torch.zeros(food_inputs.size(0), dtype=torch.long).to(device)
        optimizer.zero_grad()
        outputs = model(food_inputs)
        loss = criterion(outputs, food_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    for nonfood_inputs, _ in nonfood_train_loader:
        nonfood_inputs = nonfood_inputs.to(device)
        nonfood_labels = torch.ones(nonfood_inputs.size(0), dtype=torch.long).to(device)
        optimizer.zero_grad()
        outputs = model(nonfood_inputs)
        loss = criterion(outputs, nonfood_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (len(food_train_loader) + len(nonfood_train_loader))}')

# Validation Loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in food_val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.zeros(labels.size(0), dtype=torch.long).to(device)).sum().item()

    for images, labels in nonfood_val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.ones(labels.size(0), dtype=torch.long).to(device)).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')

# Save The Model
torch.save(model.state_dict(), "food_classification_model.pth")