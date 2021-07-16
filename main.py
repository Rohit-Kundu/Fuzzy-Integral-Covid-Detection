import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from probability_extraction import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, required = True, help='Directory where data is stored')
parser.add_argument('--epochs', type=int, default = 25, help='Number of epochs to run the models')
args = parser.parse_args()

data_dir = args.data_directory


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=10)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

#Get probability distributions from the 4 models
num_epochs = args.epochs


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)


model = models.vgg11_bn(pretrained = True)
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_vgg11')
get_probability(image_datasets,model,data_dir,model_name='Kaggle_vgg11')


model = models.googlenet(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_googlenet')
get_probability(image_datasets,model,data_dir,model_name='Kaggle_googlenet')


model = models.squeezenet1_1(pretrained = True)
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
model = model.to(device)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_squeezenet')
get_probability(image_datasets,model,data_dir,model_name='Kaggle_squeezenet')


model = models.wideresnet_50_2(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_wideresnet')
get_probability(image_datasets,model,data_dir,model_name='Kaggle_wideresnet')


#Perform the ensemble
from ensemble import *

prob1,labels = getfile("Kaggle_vgg11",root = data_dir)
prob2,_ = getfile("Kaggle_squeezenet",root = data_dir)
prob3,_ = getfile("Kaggle_googlenet",root = data_dir)
prob4,_ = getfile("Kaggle_wideresnet",root = data_dir)

ensemble_sugeno(labels,prob1,prob2,prob3,prob4)
