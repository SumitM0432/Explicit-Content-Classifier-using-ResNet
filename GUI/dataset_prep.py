import torch
from pathlib import Path
from PIL import Image

class ECCDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, transforms=None):
        super().__init__()
        
        self.transforms = transforms
        self.imgs = sorted(list(Path(root_path).glob('*.jpg')))
        self.imgs.extend(sorted(list(Path(root_path).glob('*.png'))))
        self.imgs.extend(sorted(list(Path(root_path).glob('*.jpeg'))))
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        
        label = 0
        img = self.transforms(img)
        
        return img, label