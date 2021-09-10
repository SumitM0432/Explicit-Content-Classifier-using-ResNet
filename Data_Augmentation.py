import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

transform_data = transforms.Compose([
    transforms.RandomRotation(degrees = 30),
    transforms.ColorJitter(brightness = 0.40),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomGrayscale(p = 0.2),
    transforms.ToTensor()
])
def data_aug(genre, times):

    dataset = datasets.ImageFolder('/media/levi/OS/Users/sumit/Desktop/Explicit Content Classifier/ss', transform = transform_data)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)

    # // Checking the images
    # for i in data_loader:
    #     image, target = i
    #     plt.imshow(np.transpose(image[0].numpy(), (1, 2, 0)))
    #     plt.show()

    image_num = 1
    for _ in range (times):
        for batch in  data_loader:
            img, i = batch
            save_image(img, 'Augmented Images/' + genre + '/' + genre + '_image_Aug_' + str(image_num) + '.jpg')
            image_num+=1


# data_aug('Hentai', 6)

                               