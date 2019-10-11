from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
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
    data_augmentation = args.data_augmentation 
    train_gen = load_data(args.train_path, batch_size=args.batch_size, augmentation=data_augmentation)
    valid_gen = load_data(args.valid_path, batch_size=args.batch_size)
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)

    global_step = 0
    acc = []
    for iteration in range(args.epochs):
        print('Epoch {}'.format(iteration))
        for data_batch in train_gen:           
            p_y = model(data_batch[0].to(device))
            actual = data_batch[1].to(device)
            
            ## Update weights using the optimizer calculcated gradients
            optimizer.zero_grad()
            l = loss(p_y, actual)
            l.backward()
            optimizer.step()  
            
            train_logger.add_scalar('loss', l, global_step=global_step)
            acc.extend([accuracy(p_y, actual).detach().cpu().numpy()])
            global_step += 1
        train_logger.add_scalar('accuracy', np.mean(acc), global_step=iteration)
        
        #Validate
        acc = []
        for data_batch in valid_gen:
            p_y = model(data_batch[0].to(device))
            actual = data_batch[1].to(device)
            
            l = loss(p_y, actual)
            acc.extend([accuracy(p_y, actual).detach().cpu().numpy()])
        valid_logger.add_scalar('accuracy', np.mean(acc), global_step=iteration)
        print('Accuracy {}'.format(np.mean(acc)))
        scheduler.step(np.mean(acc))
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
    parser.add_argument('-aug', '--data_augmentation', type=bool, default=False) 
    parser.add_argument('-cl','--clear_cache', type=bool, default=False)

    args = parser.parse_args()
    train(args)
