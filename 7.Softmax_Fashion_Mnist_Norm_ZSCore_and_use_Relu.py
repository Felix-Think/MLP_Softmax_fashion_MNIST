import torch 
from torchvision.datasets import FashionMNIST
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

#|%%--%%| <16Tl9KGCE9|JgL3yrt0Cw>

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

#|%%--%%| <JgL3yrt0Cw|BwezbHbXjE>

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.FashionMNIST(root='data',
                                            train=True,
                                            transform=transform,
                                            download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4)

mean = 0.0 
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1) # convert from (batch, 1, 28, 28) to (batch, 1, 784)
    mean += images.mean(2).sum(0) # 2 is the dimension to sum over
mean = mean / len(loader.dataset)

variance = 0.0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1) 
    variance += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2]) # [0, 2] means sum over the batch and pixels
std = torch.sqrt(variance / (len(loader.dataset) * 28 * 28)) # len(loader.dataset) is the number of images

print(mean, std)

#|%%--%%| <BwezbHbXjE|PoQXgPh68p>

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((mean,), (std, ))])
train_set = torchvision.datasets.FashionMNIST(root = 'data',
                                              train = True,
                                              download = True,
                                              transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 1024, num_workers = 4, shuffle = True)

test_set = torchvision.datasets.FashionMNIST(root = 'data',
                                             train = False,
                                             download = True,
                                             transform = transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1024, num_workers = 4, shuffle = False)


images, _ = train_set[0]
print(images.shape)

#|%%--%%| <PoQXgPh68p|GrnCxPMMo1>

#Function to display images
from matplotlib import pyplot as plt
import numpy as np
def show_images(image):
    img = image.numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)


for i, (images, labels) in enumerate(train_loader, 0):
    show_images(torchvision.utils.make_grid(images[:8]))
    break

#|%%--%%| <GrnCxPMMo1|DiNY4f06iJ>

# Create a model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

model.to(device)
print(model)

#|%%--%%| <DiNY4f06iJ|Bhh0jDxfUp>

input_tensor = torch.randn(5, 1, 28, 28)
output = model(input_tensor)
print(output.shape)


#|%%--%%| <Bhh0jDxfUp|xAZ1G6kHcH>
r"""°°°
# Define loss, optimizer, and evaluation function
°°°"""
#|%%--%%| <xAZ1G6kHcH|y1Ykv9vLyv>

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

def evaluate(mode, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    total = 0
    Correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader, 0):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Determine the accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            Correct += (predicted == labels).sum().item()

    test_loss =  test_loss / len(test_loader)
    accuracy = 100 * Correct / total
    return test_loss, accuracy


#|%%--%%| <y1Ykv9vLyv|WueXmVKWca>

test_loss, accuracy = evaluate(model, test_loader, criterion)
print(f'Loss: {test_loss}, Accuracy: {accuracy}')


#|%%--%%| <WueXmVKWca|fF82AWTzhi>

# Training the model
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
max_epoch = 150

for epoch in range(max_epoch):
    running_loss = 0.0
    running_accuracy = 0
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader, 0):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Determine class predictions and track accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_accuracy = 100 * correct / total
        
        # Backward and optimize
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / (i + 1)
    epoch_accuracy = 100 * correct / total
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    print(f"Epoch [{epoch + 1}/{max_epoch}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    train_losses.append(running_loss)
    train_accuracies.append(running_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
#|%%--%%| <fF82AWTzhi|hrD0TXMbUm>

plt.plot(train_losses, label = 'Train Loss')
plt.plot(test_losses, label = 'Test Loss')
plt.legend()
plt.show()

#|%%--%%| <hrD0TXMbUm|JVyHSXpLr1>

plt.plot(train_accuracies, label = 'Train Accuracy')
plt.plot(test_accuracies, label = 'Test Accuracy')
plt.legend()
plt.show()


