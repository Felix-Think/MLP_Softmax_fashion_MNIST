import torch 
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch import optim

#|%%--%%| <7xCnLNRq7Z|7o82Oms0B4>

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#|%%--%%| <7o82Oms0B4|LGqGH3pppe>

# Load data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1.0/255.0,))])

train_dataset = FashionMNIST(root ='data',
                             train = True,
                             download = True,
                             transform = transform)
train_loader = DataLoader(dataset = train_dataset, batch_size = 1024, num_workers = 4, shuffle = True)

test_dataset = FashionMNIST(root = 'data',
                            train = False,
                            download= True,
                            transform = transform)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1024, num_workers = 4, shuffle = False)
#|%%--%%| <LGqGH3pppe|B7UIq88rWy>

#Test
img, _ = train_dataset[0]
#Plot by matplotlib
from matplotlib import pyplot as plt
import numpy as np

img = img.numpy()
img = np.transpose(img, (1, 2, 0))
plt.imshow(img)

#|%%--%%| <B7UIq88rWy|pHlMCGWYMq>

# Define the function show images
def show_images(image):
    fig = plt.figure(figsize = (9, 9))
    img = image.numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
for i, (images, labels) in enumerate(train_loader, 0):
    plt.axis('off')
    show_images(torchvision.utils.make_grid(images[:10]))
    break




#|%%--%%| <pHlMCGWYMq|591ckxfhLD>
r"""°°°
# DEFINE THE MODEL
°°°"""
#|%%--%%| <591ckxfhLD|Gh8W9myhNE>

model = nn.Sequential( nn.Flatten(), 
                       nn.Linear(784, 10)) # 28x28 = 784

model = model.to(device)
print(model)

#|%%--%%| <Gh8W9myhNE|G3YCG34XnT>

#Initialize the weights
inputs_tensor = torch.rand(5, 28, 28).to(device)
outputs = model(inputs_tensor)
print(outputs.shape)




#|%%--%%| <G3YCG34XnT|UN9XwEulTE>
r"""°°°
# LOSS, OPTIMIZER, EVALUATIOn FUNCTION
°°°"""
#|%%--%%| <UN9XwEulTE|sVZgO2LDtm>

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.00001)


#|%%--%%| <sVZgO2LDtm|WAaLhyY2gH>

# Evaluate the model
def evaluation(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            total  += labels.size(0)

            # Calculate the accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_loss = test_loss / len(test_loader)
    return test_loss, accuracy


#|%%--%%| <WAaLhyY2gH|W8B7bdej3C>

test_loss, accuracy = evaluation(model, test_loader, criterion)
print(f'Test loss: {test_loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

#|%%--%%| <W8B7bdej3C|wKJgyO7gOj>

#Train model
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []
max_epochs = 100

for epoch in range(max_epochs):
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader, 0):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        #Determine the predicted classses
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        running_corrects += (predicted == labels).sum().item()
        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_accuracy = 100 * running_corrects / total
    epoch_loss = running_loss / (i + 1)
    test_loss, test_accuracy = evaluation(model, test_loader, criterion)
    print(f'Epoch: {epoch + 1}/{max_epochs}: Train Loss: {epoch_loss:.4f} Train Accuracy: {epoch_accuracy:.4f} Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.4f}')

    #save the plot
    train_accuracies.append(epoch_accuracy)
    test_accuracies.append(test_accuracy)
    train_losses.append(epoch_loss)
    test_losses.append(test_loss)

#|%%--%%| <wKJgyO7gOj|T62TO9b6Dg>

#Plot train and test losses
plt.plot(train_losses, label = 'Train loss')
plt.plot(test_losses, label = 'Test loss')
plt.legend()
plt.show()


#|%%--%%| <T62TO9b6Dg|hP54XSFfUY>

#Plot train and test accuracies
plt.plot(train_accuracies, label = 'Train accuracy')
plt.plot(test_accuracies, label = 'Test accuracy')
plt.legend()
plt.show()

