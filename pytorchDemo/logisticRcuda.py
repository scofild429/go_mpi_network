import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import pathlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import random_split # Comments

class LogisticRegression(torch.nn.Module):
    """
    Logistic regression model
    inherits the torch.nn.Module which is the base class 
    for all neural network modules.
    """
    def __init__(self, input_dim, output_dim):
        """ Initializes internal Module state. """
        super(LogisticRegression, self).__init__()
        # TODO define linear layer for the model
        self.flatten = torch.nn.Flatten()
        self.fc_layers = torch.nn.Linear(
            in_features = input_dim, # input_dim = 28 * 28 because each single image in MNIST has 28 * 28 = 784 pixels
                 # These pixels account for features of the image.
            out_features = output_dim) # output_dim = 10 because we have 10 classes, 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9.

    def forward(self, x):
        """ Defines the computation performed at every call. """
        # What are the dimensions of your input layer?
        # TODO flatten the input to a suitable size for the initial layer
        #x = x.flatten(28 * 28) # [https://www.youtube.com/watch?v=fCVuiW9AFzY&t=7s&ab_channel=deeplizard]
        x = self.flatten(x)
        # 28 * 28 = 784 We want to flatten the 2 dimensional 28 * 28 matrix into a 1 dimensional 28 * 28 = 784 array.
        # TODO run the data through the layer
        outputs = self.fc_layers(x)
        return outputs

        """ Defines the computation performed at every call. """
        # What are the dimensions of your input layer?
        # TODO flatten the input to a suitable size for the initial layer
        #x = x.flatten(28 * 28) # [https://www.youtube.com/watch?v=fCVuiW9AFzY&t=7s&ab_channel=deeplizard]
        x = self.flatten(x)
        # 28 * 28 = 784 We want to flatten the 2 dimensional 28 * 28 matrix into a 1 dimensional 28 * 28 = 784 array.
        # TODO run the data through the layer
        outputs = self.fc_layers(x)
        return outputs

def accuracy(correct, total): 
    """
    function to calculate the accuracy given the
        correct: number of correctly classified samples
        total: total number of samples
    returns the ratio
    """
    ratio = correct / total
    return ratio

def ModelTraining():
  train_loss = []
  train_acc = 0
  model.train()
  length = 0
  
  for i, (images, labels) in enumerate(train_dataloader):
    # delete the gradients from last training iteration
    optimizer.zero_grad()

    images = images.to("cuda")
    # Forward pass: get predictions
    y_pred = model(images)

    # Compute loss
    labels = labels.to("cuda")
    loss = loss_function(y_pred, labels)
        
    # Backward pass -> calculate gradients, update weights
    loss.backward() # calculate gradients
    optimizer.step() # update weights
        
    train_loss.append(loss.item()) # to translate the loss to a python float
    train_acc += (y_pred.argmax(dim = -1) == labels).int().sum().item() # Not used: y_pred[i].argmax(): accuracy of each image in batch, in each batch
    length += len(y_pred)
    
  # Returns    
  mean_loss = sum(train_loss) / len(train_loss)
  mean_acc = accuracy(train_acc, length)
  return round(mean_loss, 3), round(mean_acc, 3)


def ModelValidation():
  val_loss = []
  val_acc = 0
  length = 0
  model.eval()
  
  for i, (images, labels) in enumerate(val_dataloader):
    # delete the gradients from last training iteration
    optimizer.zero_grad()

    images = images.to("cuda")
    # Forward pass: get predictions
    y_pred = model(images)

    # Compute loss
    labels = labels.to("cuda")
    loss = loss_function(y_pred, labels)
        
    # Backward pass -> calculate gradients, update weights
    loss.backward() # calculate gradients
    optimizer.step() # update weights
        
    val_loss.append(loss.item()) # to translate the loss to a python float
    val_acc += (y_pred.argmax(dim = 1) == labels).int().sum().item()
    length += len(y_pred)
    
  # Returns    
  mean_loss = sum(val_loss) / len(val_loss)
  mean_acc = accuracy(val_acc, length)
  return round(mean_loss, 3), round(mean_acc, 3)

def RunTraining(num_epochs):
    allEpochsData = []
    for epoch in range(num_epochs):
        train_losses, train_accs = ModelTraining()
        val_losses, val_accs = ModelValidation()

        epochData = [train_losses, val_losses, train_accs, val_accs]
        print("Epoch ", epoch, "************************************")
        print("TL: ", train_losses, ", VL: ", val_losses, ", TA: ", train_accs, ", VA: ", val_accs)
    
        allEpochsData.append(epochData)
    return allEpochsData

def Plot(data, epochs_count, train_data_index, val_data_index, plt_type):
  data = np.array(data) # because if it's a tuple, we can't write epochs_data[:, 0] 
  xaxis = np.arange(epochs_count)

  plt.figure(figsize = (7, 5))
  plt.plot(xaxis, data[:, train_data_index], '--', marker = 'o', label = 'Train results') # label is for legend
  plt.plot(xaxis, data[:, val_data_index], '--', marker = 'o', label = 'Validation results') # label is for legend
  plt.xlabel('Epoch')
  if plt_type == 'l':
    plt.ylabel('Loss')
    plt.title('Epochs vs. Loss')
  else:
    plt.ylabel('Accuracy')
    plt.title('Epochs vs. Accuracy')
  plt.legend()
  plt.show()


print( torch.cuda.is_available(), torch.backends.cudnn.is_available(), torch.cuda.device_count())
data_dir = pathlib.Path('data/')
mnist = datasets.MNIST(data_dir, download=True, train=True)
X_sample, y_sample = mnist[0]
print("Data: ", X_sample)
print("Label: ", y_sample)


mnist = datasets.MNIST(data_dir, download=True, train=True, transform=transforms.ToTensor())
tmp_dataloader = torch.utils.data.DataLoader(mnist, batch_size=len(mnist), shuffle=True)
img_item = next(iter(tmp_dataloader))
img_data = img_item[0]
std, mean = torch.std_mean(img_data)
print(f'mean: {mean:.4f}, std: {std:.4f}')
mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
mnist_train = datasets.MNIST(data_dir, download=True, train=True, transform=mnist_transforms)
mnist_test = datasets.MNIST(data_dir, download=True, train=False, transform=mnist_transforms)
train_size = len(mnist_train)
test_size = len(mnist_test)
print("Train Size: ", train_size)
print("Test Size: ", test_size)
val_size = int(train_size * 0.1) # calculated before updating train_size in the following line
train_size = int(train_size * 0.9)
mnist_train, mnist_val = random_split(mnist_train, [train_size, val_size])
# TODO create dataloader for training, validation and test
batch_size = 256
train_dataloader = DataLoader(mnist_train, batch_size, shuffle = True)
val_dataloader = DataLoader(mnist_val, batch_size, shuffle = True)
test_dataloader = DataLoader(mnist_test, batch_size, shuffle = True)
x, y = next(iter(train_dataloader))
print("X Dimensions: ", x.size())
print("Y Dimensions: ", y.size())




use_cuda = True
use_cuda = False if not use_cuda else torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.cuda.get_device_name(device) if use_cuda else 'cpu'
print('Using device', device)
#print("X Dimensions: ", x)
epochs = 10
input_dim = 28 * 28
output_dim = 10
lr = 0.001
model = LogisticRegression(input_dim, output_dim)
if use_cuda:
    print("cuda is available.")
    model = model.to("cuda")
model.train()
loss_function = torch.nn.CrossEntropyLoss() # CrossEntropyLoss already includes Softmax.
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        # delete the gradients from last training iteration
        optimizer.zero_grad()

        images = images.to("cuda")
        # Forward pass: get predictions
        y_pred = model(images)

        # Compute loss
        labels = labels.to("cuda")
        loss = loss_function(y_pred, labels)

        # Backward pass -> calculate gradients, update weights
        loss.backward() # calculate gradients
        optimizer.step() # update weights

num_epochs = 10
epochs_data = RunTraining(num_epochs)
#Plot(epochs_data, num_epochs, 0, 1, 'l')
Plot(epochs_data, num_epochs, 2, 3, 'a')
            
# if __name__ == "__main__":
#     main()
