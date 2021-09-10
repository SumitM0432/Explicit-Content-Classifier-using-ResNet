# from torch.utils import data
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import config

def tr_dataset(batch_size):

    # Transformation
    transform_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Dataloader
    training_data = datasets.ImageFolder(config.TRAINING_FILE, transform = transform_data)
    validation_data = datasets.ImageFolder(config.TRAINING_FILE, transform = transform_data)

    # print (training_data)
    # print (validation_data)
    val_split = 0.10
    length = len(training_data)

    split = int(length*val_split)
    indices = torch.randperm(length)

    train_subset = torch.utils.data.Subset(training_data, indices[split:])
    val_subset = torch.utils.data.Subset(validation_data, indices[:split])

    # print (len(train_subset))
    # print (len(val_subset))

    train_dataloader = torch.utils.data.DataLoader(dataset = train_subset, batch_size = batch_size, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(dataset = val_subset, batch_size = batch_size, shuffle = False)

    print ('Classes : ', train_dataloader.dataset.dataset.class_to_idx)
    
    return train_dataloader, val_dataloader