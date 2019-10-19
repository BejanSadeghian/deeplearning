import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data, DetectionSuperTuxDataset
from . import dense_transforms
import torch.utils.tensorboard as tb


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
    
    data_rotate = args.data_rotate
    data_flip = args.data_flip
    data_colorjitter = args.data_colorjitter
#    transformer = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(), dense_transforms.ToTensor()]) #, dense_transforms.Normalize(mean=[0.2788, 0.2657, 0.2629], std=[0.205, 0.1932, 0.2237])
    transformer = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),  dense_transforms.ToTensor(), dense_transforms.to_heatmap]) 
    valid_transformer = dense_transforms.Compose([dense_transforms.ToTensor(), dense_transforms.to_heatmap]) 
    
    train_gen = load_detection_data(args.train_path, batch_size=args.batch_size, transform=transformer)
    valid_gen = load_detection_data(args.valid_path, batch_size=args.batch_size, transform=valid_transformer) #, dense_transforms.Normalize(mean=[0.2788, 0.2657, 0.2629], std=[0.205, 0.1932, 0.2237])]

    weight_tensor = torch.tensor([1-0.52683655, 1-0.02929112, 1-0.4352989, 1-0.0044619, 1-0.00411153]).to(device)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum)
#    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)
    
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
#            print(data_batch[0][0])
#            print(data_batch[1][0])
#            input('stop')
            
#            print(data_batch[0][0])
#            print(data_batch[1][0])
            p_y = model(data_batch[0].to(device))
            actual = data_batch[1].to(device)
            
#            print(p_y)
#            print(actual)

            ## Update weights using the optimizer calculcated gradients
            optimizer.zero_grad()
            l = loss(p_y, actual.float())
            l.backward()
            optimizer.step()
            
            train_logger.add_scalar('loss', l, global_step=global_step)
#            metric.add(p_y.argmax(1), actual)
            global_step += 1
        
        for i, layer in enumerate(model.parameters()):
            if layer.requires_grad:
                train_logger.add_histogram('layer {}'.format(i), layer.cpu(), global_step=iteration)
#        train_logger.add_scalar('accuracy', metric.global_accuracy, global_step=iteration)
#        train_logger.add_scalar('Intersection over union', metric.iou, global_step=iteration)
        sample = model(sample_image[0].to(device))
        train_logger.add_image('Predict',sample, global_step=iteration)
        train_logger.add_image('Actual',sample_image[1].to(device), global_step=iteration)

        #Validate
        model.eval()
#        valid_metric = ConfusionMatrix(5)
        for data_batch in valid_gen:
            p_y = model(data_batch[0].to(device))
            actual = data_batch[1].to(device)
            
            l = loss(p_y, actual.float())
#            valid_metric.add(p_y.argmax(1), actual)
#        acc = valid_metric.global_accuracy
#        valid_logger.add_scalar('accuracy', acc, global_step=iteration)
#        valid_logger.add_scalar('Intersection over union', valid_metric.iou, global_step=iteration)
#        print('Accuracy {}'.format(acc))
#        scheduler.step(acc)
            
        sample = model(sample_valid_image[0].to(device))
        valid_logger.add_image('Predict',sample, global_step=iteration)
        valid_logger.add_image('Actual',sample_image[1].to(device), global_step=iteration)
        
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
