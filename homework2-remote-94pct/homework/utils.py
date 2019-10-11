
import os
import csv
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['kart', 'bonus', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, normalize=[], augment=False):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        self.augment = augment
        self.normalize = normalize
        self.dataset_path = dataset_path
        self.label_path = os.path.join(self.dataset_path,'labels.csv')
        self.label_list = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
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
        if self.augment:
            image_to_tensor = transforms.Compose([
                transforms.ColorJitter(0.5,0.5,0.5,0.5),
                #transforms.RandomCrop(),
                transforms.RandomRotation(0.45),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
                ])
        else:
            image_to_tensor = transforms.ToTensor()
        
        return (image_to_tensor(I), label_num)

def load_data(dataset_path, num_workers=0, batch_size=128, augmentation=False):
    dataset = SuperTuxDataset(dataset_path, augment=augmentation)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
