import torch
import model
import config
from torchvision import datasets
import torchvision.transforms as transforms
import metrics

def predict_resnet(file_path, DEVICE, model_name, type):

    transform_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    data = datasets.ImageFolder(file_path, transform = transform_data)
    dataloader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = False)

    classes = {0:'Drawing', 1:'Hentai', 2:'Neutral', 3:'Porn', 4:'Sexy'}
    print (classes)

    if type == 50:
        model_load = model.resnet_model_50()
        model_load.load_state_dict(torch.load(config.OUT + model_name))
    else:
        model_load = model.resnet_model_101()
        model_load.load_state_dict(torch.load(config.OUT + model_name))

    print ('Model Loaded Sucessfully \n')
    
    pred = []
    lab = []

    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Model in Evaluation mode
        model_load.eval()

        predictions = model_load(images)
        
        # Saving the Predictions and the labels for Confusion Matrix
        pred.extend(torch.max(predictions, dim=1)[1].tolist())
        lab.extend(labels.tolist())
        
        ## Correct predictions
        correct += (torch.max(predictions, dim=1)[1] == labels).sum()
        total += labels.size(0)

    return pred, lab

if __name__ == '__main__':

    # Predicting
    pred, lab = predict_resnet(config.TESTING_FILE, config.DEVICE, model_name = 'Resnet101_e5_0.0001', type = 50) # Name of the model and type i.e. resnet50 is 50 and resnet101 means 101

    # Evaluation
    metrics.metric_scores(lab, pred)



