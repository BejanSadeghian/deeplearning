import torch
import torch.utils.tensorboard as tb
import argparse

from utils import load_data
from model import Action, save_model

def getRMSE(list_preds, list_targets, idx):
    predicted = [x[idx] for x in list_pred]
    targets = [x[idx] for x in list_targets]
    return ((predicted - targets)**2).mean().sqrt()

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
        all_targets = []
        all_predictions = []
        for batch in train_data:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            all_targets.append(batch[1].cpu())

            pred = model(images)
            all_predictions.append(pred.cpu())
            l = loss(pred, labels.squeeze())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
            global_step += 1
        train_logger.add_scalar('RMSE_steer', getRMSE(all_predictions, all_targets, 0),global_step=e)
        train_logger.add_scalar('RMSE_acceleration', getRMSE(all_predictions, all_targets, 1),global_step=e)
        train_logger.add_scalar('RMSE_brake', getRMSE(all_predictions, all_targets, 2),global_step=e)
        save_model(model)

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