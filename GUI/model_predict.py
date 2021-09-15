import os
import torch
import model
import dataset_prep
import shutil
import torchvision.transforms as transforms

def predict(file_path):

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    model_load = model.resnet_model_50();
    model_load.load_state_dict(torch.load(os.getcwd() +'\\Output\\' + 'Resnet50_e2_0.001.pth'));

    dat= dataset_prep.ECCDataset(file_path, transforms = transform_data)
    data_loader = torch.utils.data.DataLoader(dat, batch_size = 1)

    classes = {0:'Drawing', 1:'Hentai', 2:'Neutral', 3:'Porn', 4:'Sexy'}
    
    try:
        path_nsfw = os.path.join(file_path, 'NSFW')
        os.mkdir(path_nsfw)
    except:
        pass
    
    try:
        path_sfw = os.path.join(file_path, 'SFW')
        os.mkdir(path_sfw)
    except:
        pass
    count_nsfw = 0
    count_sfw = 0
    for i, (images, _) in enumerate(data_loader):

        images = images.to(DEVICE)

        model_load.eval()

        predictions = model_load(images)
        # print (predictions)
        path_ = data_loader.dataset.imgs[i]
        # break
        
        cls = classes[int(torch.max(predictions, dim=1)[1].cpu())]
        if cls in ['Porn', 'Sexy', 'Hentai']:
            shutil.move(path_, file_path + '\\NSFW\\' + os.path.basename(path_))
            count_nsfw+=1
        else:
            shutil.move(path_, file_path + '\\SFW\\' + os.path.basename(path_))            
            count_sfw+=1
    return count_nsfw, count_sfw

# if __name__ == '__main__':

#     # Predicting
#     predict('C:\\Users\\sumit\\Desktop\\WALLPAPERS')



