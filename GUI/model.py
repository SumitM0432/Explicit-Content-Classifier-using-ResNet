import torch
from torchvision import models
import torch.nn as nn
from torchsummary import summary

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# preprocessing done in dataset_prep
def resnet_model_50():

    resnet = models.resnet50(pretrained = True)

    # Already Trained Parameters will not train
    for param in resnet.parameters():
        param.requires_grad = False

    in_features = resnet.fc.in_features

    # Changing the last layer according to the classes
    fc = nn.Linear(in_features = in_features, out_features = 5)
    resnet.fc = fc

    summary(resnet.to(DEVICE), input_size = (3, 224, 224))

    return (resnet)

def resnet_model_101():

    resnet = models.resnet101(pretrained = True)

    # Already Trained Parameters will not train
    for param in resnet.parameters():
        param.requires_grad = False

    in_features = resnet.fc.in_features

    # Changing the last layer according to the classes
    fc = nn.Linear(in_features = in_features, out_features = 5)
    resnet.fc = fc

    summary(resnet.to(DEVICE), input_size = (3, 224, 224))

    return (resnet)
