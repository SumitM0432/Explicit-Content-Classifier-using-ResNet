import torch
from tqdm import tqdm
from time import time

def training_func(model, train_dataloader, val_dataloader, epochs, device, optimizer, criterion):
    # Training the model
    train_losses, val_losses = [], []
    steps = 0
    for epoch in tqdm(range(epochs)):
        
        # Taking the average loss or epoch loss
        correct_train = 0
        total_train = 0
        running_loss = 0

        start_time = time()
        iter_time = time()

        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            steps+=1
            images = images.to(device)
            labels = labels.to(device)
                                   
            # Model
            output = model(images)
            
            loss = criterion(output, labels)

            correct_train += (torch.max(output, dim=1)[1] == labels).sum()
            total_train += labels.size(0)
            
            # setting the gradient to zero or it will start accumulating
            optimizer.zero_grad()

            # Back prop
            loss.backward()
            
            # optimizer update the parameters
            optimizer.step()

            running_loss += loss.item()

            if steps % 200 == 0:
                print (f'Epoch [{epoch+1}]/[{epochs}]. Batch [{i+1}]/[{len(train_dataloader)}].', end = ' ')
                print (f'train loss {running_loss/steps:.3f}.', end = ' ')
                print (f'train_acc {(correct_train/ total_train) * 100:.3f}.', end = ' ')

                with torch.no_grad():
                    model.eval()
                    correct_val, total_val = 0, 0
                    val_loss = 0

                    for images, labels in val_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        output = model(images)
                        loss = criterion(output, labels)
                        val_loss += loss.item()

                        correct_val += (torch.max(output, dim=1)[1] == labels).sum()
                        total_val += labels.size(0)

                print(f'Val loss {val_loss / len(val_dataloader):.3f}. Val acc {correct_val / total_val * 100:.3f}.', end=' ')
                print(f'Took {time() - iter_time:.3f} seconds')
                iter_time = time()

                train_losses.append(running_loss / total_train)
                val_losses.append(val_loss / total_val)

        print(f'Epoch took {time() - start_time}') 
        # scheduler.step(average_loss)

    return model, train_losses, val_losses