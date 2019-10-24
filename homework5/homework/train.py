import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot

#Bejan imported
import random

def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    model = TCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_'+args.log_suffix), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_'+args.log_suffix), flush_secs=1)

    """
    Your code here, modify your code from prior assignments
    Hint: SGD might need a fairly high learning rate to work well here

    """
    
    ##Data
    train_data = SpeechDataset(args.train_path+'/train.txt', transform=one_hot)
    valid_data = SpeechDataset(args.valid_path+'/valid.txt', transform=one_hot)
    
    ##Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    ##Loss
    loss = torch.nn.CrossEntropyLoss()
    
    #Data index
    data_ind = list(range(len(train_data)))
    batch_size = args.batch_size
    epochs = args.epochs
    global_step = 0
    
    for e in range(epochs):
        random.shuffle(data_ind)
        n_batches = len(data_ind) // batch_size
        for b in n_batches:
            train_ind = data_ind[b * batch_size : (b+1) * batch_size]
            train_data = train_data[train_ind,:,:-1]
            train_label = train_data[train_ind,:,1:]
            
            o = model(train_data)
            
            loss = loss(o, train_label)
            l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            train_logger.add_scalar('Loss', l, global_step = global_step)
            
            global_step += 1
            
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


    args = parser.parse_args()
    train(args)
