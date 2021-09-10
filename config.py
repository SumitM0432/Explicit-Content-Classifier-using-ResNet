import torch
import numpy as np

torch.manual_seed(1)
np.random.seed(300)
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAINING_FILE = '/media/levi/OS/Users/sumit/Desktop/Explicit Content Classifier/Input/train/'
TESTING_FILE = '/media/levi/OS/Users/sumit/Desktop/Explicit Content Classifier/Input/test/'
# TRAINING_FILE = 'C:\\Users\\sumit\\Desktop\\Explicit Content Classifier\\Input\\train'
# VALIDATION_FILE = 'C:\\Users\\sumit\\Desktop\\Explicit Content Classifier\\Input\\test'
OUT = '/media/levi/OS/Users/sumit/Desktop/Explicit Content Classifier/Output/'
LEARNING_RATE = 0.0001
N_CLASSES = 5
