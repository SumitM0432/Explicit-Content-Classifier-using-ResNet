import torch
import config
import engine
import dataset_prep
import torch.nn as nn
import model
import metrics
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # DataLoaders
    train_loader, val_loader = dataset_prep.tr_dataset(batch_size = config.BATCH_SIZE)

    classes = val_loader.dataset.dataset.class_to_idx
    print (classes)
    
    # Model
    # model_load = model.resnet_model_50()
    model_load = model.resnet_model_101()
    model_load.to(config.DEVICE)
    print ('Model Loaded ---------- \n')

    # # Loss, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_load.parameters(), lr = config.LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    print ('Training Started ---------- \n')
    trained_model, train_losses, val_losses = engine.training_func(model_load, train_loader, val_loader, config.EPOCHS, config.DEVICE, optimizer, criterion)
    
    torch.save(trained_model.state_dict(), config.OUT + 'resnet101_e5_0.0001.pth')
    print ('Model Saved ---------- \n')
    
    # Plotting the training and validation loss 
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    
