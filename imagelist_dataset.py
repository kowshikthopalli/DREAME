import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
class myImageList(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.images_labels_df.iloc[idx, 0])


        image = pil_loader(img_name)
        label = self.images_labels_df.iloc[idx, 1]
    
        
        if self.transform:
            image= self.transform(image)

        return image,label

if __name__ =='__main__':
    root='/media/kowshik/Data11/DomainBed/'
    csv_file = 'PACS_splits/seed_12/domain_0.csv'
    transform= transforms.ToTensor()
    dset = myImageList(csv_file,root,transform)
    dloader = DataLoader(dset,batch_size=2)
    img,label= next(iter(dloader))
    print(img.shape,label.shape)

