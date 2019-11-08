from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import torch.nn.functional as F
import numpy as np
from .utils import load_data
from . import dense_transforms

class PositionLoss(torch.nn.Module):
    def __init__(self, image_size=(128.0,96.0)):
        super(PositionLoss, self).__init__()
        self.image_size = image_size # W, H
        self.center = torch.tensor(image_size, dtype=torch.float) / 2.0
    
    def forward(self, input, target):
        ## input (w,h)
        ## target (w,h)
        weight = torch.stack(((torch.abs(target[:,0] - self.center[0]) / self.center[0]) + 1.0, torch.ones(target.shape[0])), dim=1) 
        # weight = torch.tensor([1,0.1], dtype=torch.float)
        pl = (((input - target) * weight)**2).mean()
        return pl
            
def train(args):
    print(args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    if device == 'cuda' and args.clear_cache:
        torch.cuda.empty_cache()
    
    from os import path
    try:
        layers = eval(args.layers)
    except:
        layers = eval(args.layers[0])
    model = Planner(layers=layers).to(device)
    
    train_logger, valid_logger = None, None

    if args.log_dir is not None:
        log_suffix = args.log_suffix
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_{}'.format(log_suffix)), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_{}'.format(log_suffix)), flush_secs=1)


    """
    Your code here, modify your HW4 code
    
    """
    transformer = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0.5), dense_transforms.ToTensor()])  #, dense_transforms.ColorJitter()
    
    train_data = load_data(args.train_path, batch_size=args.batch_size, transform=transformer)
    valid_data = load_data(args.valid_path, batch_size=args.batch_size)
    
    if args.loss == 'MSE':
        loss = torch.nn.MSELoss()
    else:
        loss = PositionLoss()
    
    if arg.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum = args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
    
    global_step = 0
    
    for epoch in range(args.epochs):
        print('epoch {}'.format(epoch))
        train_error = []
        model.train()
        for batch in train_data:
            batch_data = batch[0].to(device)
            batch_label = batch[1].to(device)
            
            pred = model(batch_data)
            
            optimizer.zero_grad()
            l = loss(pred.cpu(), batch_label.cpu())
            l.backward()
            optimizer.step()
            
            error = ((pred - batch_label)**2).cpu() #For RMSE
            train_error.append(error)
            
            train_logger.add_scalar('loss', l, global_step = global_step)
            
            batch_data.to('cpu')
            batch_label.to('cpu')
            global_step += 1
            torch.cuda.empty_cache()
        rmse = torch.cat(train_error).mean().sqrt()
        train_logger.add_scalar('RMSE', rmse, global_step = epoch)
        
        valid_error = []
        model.eval()
        print('eval')
        with torch.no_grad():
            for batch in valid_data:
                batch_data = batch[0].to(device)
                batch_label = batch[1].to(device)
                
                pred = model(batch_data)
                
                error = ((pred - batch_label)**2).cpu() #For RMSE
                valid_error.append(error)
                batch_data.to('cpu')
                batch_label.to('cpu')
                torch.cuda.empty_cache()
            
        rmse = torch.cat(valid_error).mean().sqrt()
        valid_logger.add_scalar('RMSE', rmse, global_step = epoch)
        scheduler.step(rmse)
        save_model(model, args.model_label)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-t', '--train_path', type=str)
    parser.add_argument('-v', '--valid_path', type=str)
    parser.add_argument('-ep', '--epochs', type=int, default=500)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-mom', '--momentum', type=float, default=0.9)
    parser.add_argument('-ls', '--log_suffix', type=str, default='')
    parser.add_argument('-cl','--clear_cache', type=bool, default=True)
    parser.add_argument('-rot', '--data_rotate', type=bool, default=False)
    parser.add_argument('-flip', '--data_flip', type=bool, default=False)
    parser.add_argument('-jit', '--data_colorjitter', type=bool, default=False)
    parser.add_argument('-la', '--layers', type=str, default='[32,32,64]')
    parser.add_argument('-ml', '--model_label', type=str)
    parser.add_argument('-lo', '--loss', type=str, default='MSE')
    parser.add_argument('-op', '--optim', type=str, default='SGD')

    args = parser.parse_args()
    train(args)
