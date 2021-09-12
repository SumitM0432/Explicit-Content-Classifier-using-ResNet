# Explicit-Content-Classifier-using-ResNet

## Objective
Nowadays, The amount of explicit content available on the internet is ginormous and this type of content can be strayed and uploaded to websites that do not endorse these or some sites that are not 18+, hence classifying this type of content is really of paramount significance.

The objective of this project is to classify explicit content that contains inappropriate images like pornography and Hentai. The classifier used for this is ResNet50 and ResNet101 also known as Residual Neural Network. There are five categories that the model is trained on which are Porn, Hentai, Sexy, Drawing, and Neutral. Porn, Hentai, and Sexy can be classified as NSFW **(Not Safe For Work)** further and the other two are SFW **(Safe For Work)**.

## Dataset
The dataset is taken from various sources like [Kaggle](https://www.kaggle.com/drakedtrex/my-nsfw-dataset), [Internet Archive](https://archive.org/details/NudeNet_classifier_dataset_v1), and [Github](https://github.com/alex000kim/nsfw_data_scraper/tree/master/raw_data). The dataset sources contained explicit images of different genre, hence they are picked according to the used classes. The final dataset contained around 170,000 images from which 10,000 images are used for testing (2000 per class). As for Validation, from 160,000 images about 10% are used for validation.

The data augmentation is also done on Drawings and Hentai classes as the number of images was on the lower side. For data augmentation, 4 attributes were changed randomly that are brightness, horizontal flip, grayscale, and rotation of 30 degrees.

## Files
- config.py - Configuration File
- Data_Augmentation.py - Contains code for data augmentation as explained above.
- Split.py - Used to split the dataset into train and test.
- dataset_prep.py - Preprocessing is done in the file.
- engine.py - Contains the training function for the model [training and validation]
- metrics.py - Contains the metrics for evaluation like confusion matrix plot
- model.py - contains the PyTorch Models -- Resnet50 and Resnet101
- predict.py - Use the model to predict on Test Dataset.
- train.py - main run file

## Preprocessing
For preprocessing, the images are resized to 224x224 as the model input size was the same and the images were normalized as well (given in the `data_prep.py`) and finally converted to a Tensor.

## Model
Two models were trained named ResNet50 and ResNet101 from the [Torchvision Model](https://pytorch.org/vision/stable/models.html) library. Accuracy, classification report, and confusion metrics are used to judge the performances of the model. Each model is trained for 2 and 5 epochs with 0.001 and 0.0001 Learning rates. The loss curve and confusion matrix plotted are given in the `Images` folder.
> The framework used for models is PyTorch.
> 
