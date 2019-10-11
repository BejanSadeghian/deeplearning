from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import numpy as np


def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    if device == 'cuda' and args.clear_cache:
        torch.cuda.empty_cache()

    model = CNNClassifier().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        log_suffix = args.log_suffix
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_{}'.format(log_suffix)))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid_{}'.format(log_suffix)))

    """
    Your code here, modify your HW1 code
    
    """
#    
#    mean = []
#    std = []
#    for b in load_data(args.train_path, batch_size=args.batch_size):
#        print(b[0])
#        print(b[0].mean(dim=(0,2,3)))
#        print(b[0].std(dim=(0,2,3)))
#        mean.append(b[0].mean(dim=(0,2,3)))
#        std.append(b[0].std(dim=(0,2,3)))
#    
#    t_mean = torch.mean(torch.stack(mean),0)
#    t_std = torch.sqrt(torch.mean(torch.stack(std) ** 2,0))
#    print('res')
#    print(t_mean)
#    print(t_std)
    
    data_rotate = args.data_rotate
    data_flip = args.data_flip
    data_colorjitter = args.data_colorjitter
    data_normalize = args.data_normalize

    train_gen = load_data(args.train_path, batch_size=args.batch_size, rotate=data_rotate, flip=data_flip, colorjitter=data_colorjitter, normalize=data_normalize)
    valid_gen = load_data(args.valid_path, batch_size=args.batch_size, normalize=data_normalize)
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)

    global_step = 0
    for iteration in range(args.epochs):
        print('Epoch {}'.format(iteration))
        metric = ConfusionMatrix(6)
        model.train()
        for data_batch in train_gen:           
            p_y = model(data_batch[0].to(device))
            actual = data_batch[1].to(device)
            
            ## Update weights using the optimizer calculcated gradients
            optimizer.zero_grad()
            l = loss(p_y, actual)
            l.backward()
            optimizer.step()  
            
            train_logger.add_scalar('loss', l, global_step=global_step)
#            acc.extend([accuracy(p_y, actual).detach().cpu().numpy()])
            metric.add(p_y.argmax(1), actual)
            global_step += 1
        train_logger.add_scalar('accuracy', metric.global_accuracy, global_step=iteration)
        
        #Validate
        model.eval()
        valid_metric = ConfusionMatrix(6)
        for data_batch in valid_gen:
            p_y = model(data_batch[0].to(device))
            actual = data_batch[1].to(device)
            
            l = loss(p_y, actual)
#            acc.extend([accuracy(p_y, actual).detach().cpu().numpy()])
            valid_metric.add(p_y.argmax(1), actual)
        acc = valid_metric.global_accuracy
        valid_logger.add_scalar('accuracy', acc, global_step=iteration)
        print('Accuracy {}'.format(acc))
        scheduler.step(acc)
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
    parser.add_argument('-cl','--clear_cache', type=bool, default=True)
    parser.add_argument('-rot', '--data_rotate', type=bool, default=True)
    parser.add_argument('-flip', '--data_flip', type=bool, default=True)
    parser.add_argument('-jit', '--data_colorjitter', type=bool, default=True)
    parser.add_argument('-norm', '--data_normalize', type=bool, default=True)

    args = parser.parse_args()
    train(args)
