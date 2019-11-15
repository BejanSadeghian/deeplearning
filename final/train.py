import torch
import torch.utils.tensorboard as tb
import argparse

from utils import load_data
from model import Action

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)
    model = Action().to(device)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_data = load_data(args.train_path, batch_size=args.batch_size)
    valid_data = load_data(args.valid_path, batch_size=args.batch_size)

    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=args.logdir, flush_secs=1, filename_suffix='train_{}'.format(args.log_suffix))
        valid_logger = tb.SummaryWriter(log_dir=args.logdir, flush_secs=1, filename_suffix='valid_{}'.format(args.log_suffix))

    global_step = 0
    for e in range(args.epochs):
        print('Epoch:',e)
        for batch in train_data:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            # print(images.shape)

            pred = model(images)
            l = loss(pred, labels)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
            global_step += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train(args)