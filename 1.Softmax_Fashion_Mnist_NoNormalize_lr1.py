from torchvision.datasets import FashionMNIST
import torchvision
import torch
from torchvision.transforms import transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
#|%%--%%| <4G0NMdK8Vd|9z3lQSXPTg>

#Check if GPU is availale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#|%%--%%| <9z3lQSXPTg|nlGxDdQdY5>

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1.0/255.0,))]) # mean = 0, std = 1/255
# Load the FashionMNIST dataset
train_set = FashionMNIST(root = 'data',
                         train = True,
                         download = True,
                         transform = transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1024, num_workers=4, shuffle=True)
test_set = FashionMNIST(root = 'data',
                        train = False,
                        download = True,
                        transform = transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, num_workers=4, shuffle=False)

#Print a sample of the data
img, _ = train_set[0]
print(img.size())
print(type(img))


#|%%--%%| <nlGxDdQdY5|ekyofPJUeU>

from matplotlib import pyplot as plt
import numpy as np

#|%%--%%| <ekyofPJUeU|qeI7T7XRdm>


#Function to display images
def show_images(image):
    fig = plt.figure(figsize = (9, 9))
    img = image.numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)

for i, (images, labels) in enumerate(data_loader, 0):
    plt.axis('off')
    show_images(torchvision.utils.make_grid(images[:8])) # Show the first 8 images in each batch, so we can break the loop
    break


#|%%--%%| <qeI7T7XRdm|euKVpfP4rr>
r"""°°°
# MODEL
°°°"""
#|%%--%%| <euKVpfP4rr|9DkleJcujH>

# Define the model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)
model = model.to(device)
print(model)

#|%%--%%| <9DkleJcujH|NBi1LUEQR9>

# Initialize the random weights of the model

input_tensor = torch.rand(5, 28, 28).to(device) # Create a random tensor of size 5 x 28 x 28, 5 images of 28 x 28 pixels
output = model(input_tensor)

print(f"Input tensor shape: {output.shape}")

#|%%--%%| <NBi1LUEQR9|Y9qyo73IQm>
r"""°°°
# LOSS, OPTIMIZER, EVALUATION FUNCTIOn
°°°"""
#|%%--%%| <Y9qyo73IQm|lDuGz1y3rg>

# Define the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01) # SGD is stand for Stochastic Gradient Descent

#|%%--%%| <lDuGz1y3rg|nSML0OSFhL>

def evaluation(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    # in this case we don't need to calculate the gradients so we use torch.no_grad()
    with torch.no_grad():
        for images, labels in test_loader:
            #move the images and labels to the device the model is on
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() # item() is used to convert the tensor value to a python number
            
            _, predicted = torch.max(outputs.data, 1) # Get the predicted class with the highest probalility
            #print(f'Shape of predicted: {predicted.shape}')
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_loss = test_loss / len(test_loader) # Average loss

    return test_loss, accuracy
    

#|%%--%%| <nSML0OSFhL|kN8eatoQBh>


test_loss, accuracy = evaluation(model, test_loader, criterion)
print(f'Test loss: {test_loss}')
print(f'Accuracy: {accuracy}')
#|%%--%%| <kN8eatoQBh|mHOdIau8B1>

#train

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
epochs = 100

for  epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0   # to track number of correct predictions
    total = 0             # to track total number of samples    
    for i, (images, labels) in enumerate(train_loader, 0):
        #move input and labels to the device
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients

        optimizer.zero_grad()
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Determine the predicted classes
        _, predicted = torch.max(outputs, 1) # Get the predicted values with every sample
        total += labels.size(0)
        running_correct += (predicted == labels).sum().item()
        # Backward
        loss.backward()
        optimizer.step()
    epoch_accuracy = 100 * running_correct / total
    epoch_loss = running_loss / len(data_loader)
    test_loss, test_accuracy = evaluation(model, test_loader, criterion)
    print(f'Epoch: {epoch + 1} / {epochs}: Train Loss: {epoch_loss:.4f} Train Accuracy: {epoch_accuracy:.4f} Test Loss: {test_loss:4f} Test Accuracy: {test_accuracy:4f}')

    train_losses.append(epoch_loss)
    test_losses.append(test_loss)
    train_accuracies.append(epoch_accuracy)
    test_accuracies.append(test_accuracy)
        
#|%%--%%| <mHOdIau8B1|2eYBAljiga>

#Plot train and test losses

plt.plot(train_losses, label = 'train loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()

#|%%--%%| <2eYBAljiga|W9Pwu811qY>

#Plot train and test accuracies
plt.plot(train_accuracies, label = 'train accuracy')
plt.plot(test_accuracies, label = 'test accuracy')
plt.legend()
plt.show()

