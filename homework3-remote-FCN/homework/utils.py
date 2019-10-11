import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import csv
import os
import numpy as np
from . import dense_transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, **kwargs):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        self.rotate = False
        self.flip = False
        self.colorjitter = False
        self.normalize = False
        if 'rotate' in kwargs:
            print('Rotating Data')
            self.rotate = kwargs['rotate']
        
        if 'flip' in kwargs:
            print('Flip Data')
            self.flip = kwargs['flip']   
        
        if 'colorjitter' in kwargs:
            print('ColorJitter Data')
            self.colorjitter = kwargs['colorjitter']
            
        if 'normalize' in kwargs:
            self.normalize = kwargs['normalize']
            print('Normalizing Data')
            self.normalize_mean = (0.3235, 0.3310, 0.3445)
            self.normalize_std = (0.2530, 0.2222, 0.2482)
            
        self.dataset_path = dataset_path
        self.label_path = os.path.join(self.dataset_path,'labels.csv')
        self.label_list = LABEL_NAMES 
        data = []
        
        ## This "with" block code was code I wrote for my own usecase but was 
        ## based on a solution found on stack overflow as guidance.
        ## Citation: https://realpython.com/python-csv/
        with open(self.label_path) as file_obj:
            reader = csv.reader(file_obj, delimiter=',')
            line_count = 0
            for row in reader: 
                if line_count == 0:
                    data_headers = np.array(row)
                else:
                    new_line = row
                    new_line.append(row[0].split('.')[0]) #Grab the id
                    data.append(new_line)
                line_count += 1
        self.raw_data = np.array(data)
        self.data_headers = data_headers
        

    def __len__(self):
        """ Returns the size of the data
        """ 
        return(self.raw_data.shape[0])

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        image_file, label = self.raw_data[idx][:2]
        #Get the int of the label
        label_num = self.label_list.index(label)
        
        I = Image.open(os.path.join(self.dataset_path,image_file))
        image_to_tensor = transforms.ToTensor()
        if self.rotate or self.flip or self.colorjitter:
            transform_list = []
            if self.rotate:
                transform_list.append(transforms.RandomRotation(0.45))
            if self.flip:
                transform_list.append(transforms.RandomHorizontalFlip(0.5))
            if self.colorjitter:
                transform_list.append(transforms.ColorJitter(0.5,0.5,0.5,0.5))
                
            transform_list.append(transforms.ToTensor())
            if self.normalize:
                transform_list.append(transforms.Normalize(self.normalize_mean, self.normalize_std))
            image_to_tensor = transforms.Compose(transform_list)
        else:
            image_to_tensor = transforms.ToTensor()
        return (image_to_tensor(I), label_num)


class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0)

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


if __name__ == '__main__':
    dataset = DenseSuperTuxDataset('dense_data/train', transform=dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor()]))
    from pylab import show, imshow, subplot, axis

    for i in range(15):
        im, lbl = dataset[i]
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(dense_transforms.label_to_pil_image(lbl))
        axis('off')
    show()
    import numpy as np

    c = np.zeros(5)
    for im, lbl in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))
