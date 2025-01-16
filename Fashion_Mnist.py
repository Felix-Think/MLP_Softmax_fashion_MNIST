#Download image
from urllib import request
filenames = ["train-images-idx3-ubyte.gz",
             "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz",
             "t10k-labels-idx1-ubyte.gz"]
url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
folder = "datafashion_mnist/"

#|%%--%%| <qmealVMuZ7|rpNsLm8WyB>

for name in filenames:
    print("Downloading", name)
    request.urlretrieve(url + name, folder + name)
#|%%--%%| <rpNsLm8WyB|8eqUuqxR5G>

#gzip use frombuffter of numpy
import gzip
import numpy as np

with gzip.open('datafashion_mnist/train-images-idx3-ubyte.gz', 'rb') as f:
    X_train = np.frombuffer(f.read(), np.uint8, offset = 16).reshape(-1, 28*28)

with gzip.open('datafashion_mnist/t10k-images-idx3-ubyte.gz', 'rb') as f:
    X_test = np.frombuffer(f.read(), np.uint8, offset = 16).reshape(-1, 28*28)

with gzip.open('datafashion_mnist/train-labels-idx1-ubyte.gz', 'rb') as f:
    y_train = np.frombuffer(f.read(), np.uint8, offset = 8)

with gzip.open('datafashion_mnist/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    y_test = np.frombuffer(f.read(), np.uint8, offset = 8)

#|%%--%%| <8eqUuqxR5G|fmPgv7429s>


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#|%%--%%| <fmPgv7429s|XCvwbAIhfB>

import matplotlib.pyplot as plt
import random

indices = list(np.random.randint(0, X_train.shape[0], size = 144))# Create a list of 144 random indices inthe shape of the training data
rows = 12
cols = 12
fig = plt.figure(figsize=(9, 9))
# Create a 12x12 grid of subplots
for i in range(1, cols * rows + 1):
    img = X_train[indices[i - 1]].reshape(28, 28)
    fig.add_subplot(rows, cols, i)
    plt.axis('off')
    plt.imshow(img, cmap='gray')


#|%%--%%| <XCvwbAIhfB|CzRAxzwHWL>

# image if not have cmap='gray' will be colorful
indices = list(np.random.randint(0, X_train.shape[0], size = 144))
rows = 12
cols = 12
fig = plt.figure(figsize=(9, 9))

for i in range(1, cols * rows + 1):
    img = X_train[indices[i - 1]].reshape(28, 28)
    fig.add_subplot(rows, cols, i)
    plt.axis('off')
    plt.imshow(img)
#|%%--%%| <CzRAxzwHWL|MBaXDD9Noy>


# Current, file is not saved as a image file
# We will save it as a iamge file
from PIL import Image

indices = list(np.random.randint(0, X_train.shape[0], size = 144))
for i in range(144):
    img = Image.fromarray(X_train[indices[i]].reshape(28, 28))
    img.save("datafashion_mnist/image_" + str(i) + ".png")
