import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot, vocab

#Bejan imported
import random
import numpy as np

def get_nll(data_gen, model, vocab):
    ll = []
    for v in data_gen:
        in_string = [vocab[i] for i in v.argmax(dim=0)]
        in_string = ''.join(in_string)
#        print(model.predict_all(in_string))
        ll.append(float((model.predict_all(in_string)[:,:-1] * one_hot(in_string)).sum()/len(v)))
    return -np.mean(ll)

def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = TCN().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_'+args.log_suffix), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_'+args.log_suffix), flush_secs=1)

    """
    Your code here, modify your code from prior assignments
    Hint: SGD might need a fairly high learning rate to work well here

    """
    
    ##Data
    train_gen = SpeechDataset(args.train_path+'/train.txt', transform=one_hot)
    valid_gen = SpeechDataset(args.valid_path+'/valid.txt', transform=one_hot)
    
    ##Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    ##Loss
    loss = torch.nn.CrossEntropyLoss()
    
    #Data index
    data_ind = list(range(len(train_gen)))
    batch_size = args.batch_size
    epochs = args.epochs
    global_step = 0
    
    for e in range(epochs):
        print('Epoch',e)
        random.shuffle(data_ind)
        n_batches = len(data_ind) // batch_size
        for b in range(n_batches):
            train_ind = data_ind[b * batch_size : (b+1) * batch_size]
            batch = []
            for i in train_ind:
                batch.append(train_gen[i])
            batch = torch.stack(batch, dim=0).to(device)
            train_data = batch[:,:,:-1]
            train_label = batch[:,:,:].argmax(dim=1)
            
            o = model(train_data)
            l = loss(o, train_label)
            if train_logger:
                train_logger.add_scalar('Loss', l, global_step = global_step)
                
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            optimizer.step()
            
            global_step += 1
            
        nll = get_nll(train_gen, model, vocab)
        if train_logger:
            print('adding NLL')
            train_logger.add_scalar('nll', nll, global_step = e)
        
        print('validate')
        nll = get_nll(valid_gen, model, vocab)
        if valid_logger:
            valid_logger.add_scalar('nll', nll, global_step = e)
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
