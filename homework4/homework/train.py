import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from .models import Detector, save_model
from .utils import load_detection_data, DetectionSuperTuxDataset
from . import dense_transforms
import torch.utils.tensorboard as tb

## Copied this code from the test code
def point_in_box(p, x0, y0, x1, y1):
    return x0 <= p[0] < x1 and y0 <= p[1] < y1


def point_close(p, x0, y0, x1, y1, d=5):
    return ((x0 + x1 - 1) / 2 - p[0]) ** 2 + ((y0 + y1 - 1) / 2 - p[1]) ** 2 < d ** 2


def box_iou(p, x0, y0, x1, y1, t=0.5):
    iou = abs(min(p[0] + p[2], x1) - max(p[0] - p[2], x0)) * abs(min(p[1] + p[3], y1) - max(p[1] - p[3], y0)) / \
          abs(max(p[0] + p[2], x1) - min(p[0] - p[2], x0)) * abs(max(p[1] + p[3], y1) - min(p[1] - p[3], y0))
    return iou > t


class PR:
    def __init__(self, min_size=20, is_close=point_in_box):
        self.min_size = min_size
        self.total_det = 0
        self.det = []
        self.is_close = is_close

    def add(self, d, lbl):
        small_lbl = [b for b in lbl if abs(b[2] - b[0]) * abs(b[3] - b[1]) < self.min_size]
        large_lbl = [b for b in lbl if abs(b[2] - b[0]) * abs(b[3] - b[1]) >= self.min_size]
        used = [False] * len(large_lbl)
        for s, *p in d:
            match = False
            for i, box in enumerate(large_lbl):
                if not used[i] and self.is_close(p, *box):
                    match = True
                    used[i] = True
                    break
            if match:
                self.det.append((s, 1))
            else:
                match_small = False
                for i, box in enumerate(small_lbl):
                    if self.is_close(p, *box):
                        match_small = True
                        break
                if not match_small:
                    self.det.append((s, 0))
        self.total_det += len(large_lbl)

    @property
    def curve(self):
        true_pos, false_pos = 0, 0
        r = []
        for t, m in sorted(self.det, reverse=True):
            if m:
                true_pos += 1
            else:
                false_pos += 1
            prec = true_pos / (true_pos + false_pos)
            recall = true_pos / self.total_det
            r.append((prec, recall))
        return r

    @property
    def average_prec(self, n_samples=11):
        max_prec = 0
        cur_rec = 1
        precs = []
        for prec, recall in self.curve[::-1]:
            max_prec = max(max_prec, prec)
            while cur_rec > recall:
                precs.append(max_prec)
                cur_rec -= 1.0 / n_samples
        return sum(precs) / len(precs)


class calc_metric(object):
    def __init__(self,model):
        # Compute detections
        self.pr_box = [PR() for _ in range(3)]
        self.pr_dist = [PR(is_close=point_close) for _ in range(3)]
        self.model = model
        
    def add(self, dataset):
        for img, *gts in dataset:
            d = self.model.detect(img)
            for i, gt in enumerate(gts):
                self.pr_box[i].add([j[1:] for j in d if j[0] == i], gt)
                self.pr_dist[i].add([j[1:] for j in d if j[0] == i], gt)
    
    def calc(self):
        ap0 = self.pr_box[0].average_prec
        ap1 = self.pr_box[1].average_prec
        ap2 = self.pr_box[2].average_prec
        apb0 = self.pr_dist[0].average_prec
        apb1 = self.pr_dist[1].average_prec
        apb2 = self.pr_dist[2].average_prec
        return (ap0, ap1, ap2, apb0, apb1, apb2)


class FocalLoss(torch.nn.Module):
    ## Focal Loss written according to https://arxiv.org/pdf/1708.02002.pdf
    ## Also borrowed the idea of epislon from https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, input, target):
        p = input.view(-1)
        p.clamp_(self.eps, 1 - self.eps)
        y = target.view(-1)
        fl = -((1-p)**self.gamma) * F.logsigmoid(p) * y - ((p)**self.gamma) * F.logsigmoid(1-p) * (1-y)
        return fl.mean()
        

def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    if device == 'cuda' and args.clear_cache:
        torch.cuda.empty_cache()
        
    model = Detector().to(device)
    train_logger, valid_logger = None, None

    if args.log_dir is not None:
        log_suffix = args.log_suffix
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_{}'.format(log_suffix)), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_{}'.format(log_suffix)), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    
#    data_rotate = args.data_rotate
#    data_flip = args.data_flip
#    data_colorjitter = args.data_colorjitter
#    transformer = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(), dense_transforms.ToTensor()]) #, dense_transforms.Normalize(mean=[0.2788, 0.2657, 0.2629], std=[0.205, 0.1932, 0.2237])
    transformer = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),  dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]) 
    valid_transformer = dense_transforms.Compose([dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]) 
    
    train_gen = load_detection_data(args.train_path, batch_size=args.batch_size, transform=transformer)
#    valid_gen = load_detection_data(args.valid_path, batch_size=args.batch_size, transform=valid_transformer) #, dense_transforms.Normalize(mean=[0.2788, 0.2657, 0.2629], std=[0.205, 0.1932, 0.2237])]
    valid_metric_dataset  = DetectionSuperTuxDataset(args.valid_path, min_size=0)
    
#    weight_tensor = torch.tensor([1-0.52683655, 1-0.02929112, 1-0.4352989, 1-0.0044619, 1-0.00411153]).to(device)
#    loss = torch.nn.BCEWithLogitsLoss()
    loss = FocalLoss(gamma=2)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum)
#    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
#    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)
    
    #Sample Image
    image_id = np.random.randint(0,100)
    train_dataset = DetectionSuperTuxDataset(args.train_path, transform=transformer)
    valid_dataset = DetectionSuperTuxDataset(args.valid_path, transform=valid_transformer)
    sample_image = train_dataset[image_id]
    sample_valid_image = valid_dataset[image_id]
    
    global_step = 0
    for iteration in range(args.epochs):
        print('Epoch {}'.format(iteration))
#        metric = ConfusionMatrix(5)
        model.train()
        for data_batch in train_gen:  
            p_y = model(data_batch[0].to(device))
            actual = data_batch[1].to(device)
            
            ## Update weights using the optimizer calculcated gradients
            optimizer.zero_grad()
            l = loss(p_y.cpu(), actual.float().cpu())
            l.backward()
            optimizer.step()
            
            train_logger.add_scalar('loss', l, global_step=global_step)
            
            global_step += 1
        
#        for i, layer in enumerate(model.parameters()):
#            if layer.requires_grad:
#                train_logger.add_histogram('layer {}'.format(i), layer.cpu(), global_step=iteration)

        im = sample_image[0].unsqueeze(0)
        sample = model(im.to(device))
        sample.squeeze_(0)
#        detection = model.detect(sample)
        train_logger.add_image('Original',sample_image[0].cpu(), global_step=iteration)
        train_logger.add_image('Heatmap',sample.cpu(), global_step=iteration)
#        train_logger.add_image('Detected',sample.cpu(), global_step=iteration)
        train_logger.add_image('Actual',sample_image[1].cpu(), global_step=iteration)

        #Validate
        print('validate')
        model.eval()
        valid_metric = calc_metric(model)
        valid_metric.add(valid_metric_dataset)
        ap0, ap1, ap2, apb0, apb1, apb2 = valid_metric.calc()
        
        #Record Valid results
        valid_logger.add_scalar('AP0', ap0, global_step=iteration)
        valid_logger.add_scalar('AP1', ap1, global_step=iteration)
        valid_logger.add_scalar('AP2', ap2, global_step=iteration)
        valid_logger.add_scalar('AP_box0', apb0, global_step=iteration)
        valid_logger.add_scalar('AP_box1', apb1, global_step=iteration)
        valid_logger.add_scalar('AP_box2', apb2, global_step=iteration)
        
        im = sample_valid_image[0].unsqueeze(0)
        sample = model(im.to(device))
        sample.squeeze_(0)
#        detection = model.detect(sample)
        
        valid_logger.add_image('Original',sample_valid_image[0].cpu(), global_step=iteration)
        valid_logger.add_image('Heatmap',sample.cpu(), global_step=iteration)
#        valid_logger.add_image('Detected',sample.cpu(), global_step=iteration)
        valid_logger.add_image('Actual',sample_valid_image[1].cpu(), global_step=iteration)
        
        train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step=iteration)
        save_model(model)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-t', '--train_path', type=str)
    parser.add_argument('-v', '--valid_path', type=str)
    parser.add_argument('-ep', '--epochs', type=int, default=500)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-mom', '--momentum', type=float, default=0.9)
    parser.add_argument('-ls', '--log_suffix', type=str, default='')
    parser.add_argument('-cl','--clear_cache', type=bool, default=False)
    parser.add_argument('-rot', '--data_rotate', type=bool, default=False)
    parser.add_argument('-flip', '--data_flip', type=bool, default=False)
    parser.add_argument('-jit', '--data_colorjitter', type=bool, default=False)


    args = parser.parse_args()
    train(args)
