import torch
import os
import csv
import numpy as np
import ast
from PIL import Image
from torchvision import transforms

class AgentData(torch.utils.data.DataLoader):

    def __init__(self, dataset_path, norm=True, recalc_norm=False):
        self.norm = norm
        self.dataset_path = dataset_path
        self.ids = [x for x in os.listdir(self.dataset_path) if x.endswith('.csv')]
        data = []
        for i in self.ids:
            with open(os.path.join(self.dataset_path, i)) as file_obj:
                reader = csv.reader(file_obj, delimiter=',')
                for ix, row in enumerate(reader):
                    if ix != 0:
                        break
                    data.append((i[:-4]+'.png', np.array([ast.literal_eval(x) if x.lower() != 'false' and x.lower() != 'true' else bool(ast.literal_eval(x)) for x in row])))
        self.data = data
        if recalc_norm:
            norm_calc = []
            for d in data:
                norm_calc.append(np.array(Image.open(os.path.join(self.dataset_path, d[0]))))
            norm_calc = np.stack(norm_calc)
            print(norm_calc.shape)
            print(torch.tensor(norm_calc, dtype=torch.float).mean([0,1,2]))
            print(torch.tensor(norm_calc, dtype=torch.float).std([0,1,2]))
        self.mean = torch.tensor([8.9478, 8.9478, 8.9478], dtype=torch.float) 
        self.std = torch.tensor([47.0021, 42.1596, 39.2562], dtype=torch.float)
        # print(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        targets = self.data[idx][1:]

        img = Image.open(os.path.join(self.dataset_path, image))
        
        if self.norm:
            image_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        else:
            image_to_tensor = transforms.ToTensor()
        img = image_to_tensor(img)
        return (img, torch.tensor(targets, dtype=torch.float))

def load_data(path_to_data, batch_size=64):
    d = AgentData(path_to_data)
    return torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    data = AgentData('/Users/bsadeghian/Documents/UTCS/Deep Learning/deeplearning/final/train_data')
    print(data[0])