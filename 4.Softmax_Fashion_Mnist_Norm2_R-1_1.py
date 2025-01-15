import torchvision
import torch
from torchvision.datasets import FashionMNIST
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#|%%--%%| <VuTAK7n6Hd|Zn3XTTyha6>

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#|%%--%%| <Zn3XTTyha6|n0uJvc03OB>

# Load data
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,), (0.5,))])
train_data = FashionMNIST(root = 'data',
                          train = True,
                          download = True,
                          transform = transform)
data_loader = DataLoader(dataset = train_data, num_workers = 4, batch_size = 1024, shuffle = True)

test_data = FashionMNIST(root = 'data',
                         train = False,
                         download = True,
                         transform = transform)
test_loader = DataLoader(dataset = test_data, num_workers = 4, batch_size = 1024, shuffle = True)


# Print 1 image
image, label = train_data[0]
print(image.shape)

#Define show image function
import matplotlib.pyplot as plt
import numpy as np
def show_image(image):
    image = image / 2.0 + 0.5
    img = image.numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap = 'gray')
    plt.show()

for i, (images, label) in enumerate(data_loader):
    show_image(torchvision.utils.make_grid(images[:8]))
    break

#|%%--%%| <n0uJvc03OB|YTco5aW1i9>

# Define model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)

model.to(device)
print(model)

#|%%--%%| <YTco5aW1i9|0AkB01GvaC>

input_tensor = torch.randn(5, 28, 28).to(device)
output = model(input_tensor)
print(output.shape)


#|%%--%%| <0AkB01GvaC|Wk1VQ6pvWN>
r"""°°°
# Define Loss, Optimizer, and evaluation function
°°°"""
#|%%--%%| <Wk1VQ6pvWN|yHJkbC7OLH>

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# Evaluation function
def evaluate(model, test_loader, criterion):
    model.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_loss = test_loss / len(test_loader)
    return test_loss , accuracy


#|%%--%%| <yHJkbC7OLH|kjnpUU0n5u>

test_loss ,accuracy = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.2f}%")




#|%%--%%| <kjnpUU0n5u|xMsFpvWbAy>
r"""°°°
#Train model
°°°"""
#|%%--%%| <xMsFpvWbAy|lgehGX8kEb>

#define paremeters
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

# Train the model
max_epochs = 100
for epoch in range(max_epochs):
    # Initialize some parameters
    running_loss = 0.0
    running_corrects = 0.0
    total = 0
    
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        #Determine class predictions and accuracy

        _, predicted = torch.max(outputs.data, 1)
        running_corrects += (predicted == labels).sum().item()
        total += labels.size(0)

        # Backward and optimize
        loss.backward()
        optimizer.step()

    epoch_accuracy = 100 * running_corrects / total
    epoch_loss = running_loss / len(data_loader)
    test_loss_epoch, test_accuracy_epoch = evaluate(model, test_loader, criterion)
    print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, Test Loss: {test_loss_epoch:.4f}, Test Accuracy: {test_accuracy_epoch:.2f}%")
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    test_loss.append(test_loss_epoch)
    test_accuracy.append(test_accuracy_epoch)

#|%%--%%| <lgehGX8kEb|THp4pAElPV>

# Plot loss and accuracy
plt.plot(train_loss, label = 'train loss')
plt.plot(test_loss, label = 'test loss')
plt.legend()
plt.show()

#|%%--%%| <THp4pAElPV|26veYTqK8L>

plt.plot(train_accuracy, label = 'train accuracy')
plt.plot(test_accuracy, label = 'test accuracy')
plt.legend()
plt.show()

