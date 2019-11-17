import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np
import pystk
from .player import HockeyPlayer

# from utils import load_data
from .model import Action, save_model

class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info):
        return self.player.act(image, player_info)

def getRMSE(list_preds, list_targets, idx):
    predicted = np.array([x[idx] for x in list_preds])
    targets = np.array([x[idx] for x in list_targets])
    return np.sqrt(((predicted - targets)**2).mean())

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)
    model = Action(normalize=True, inference=False).to(device)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    batch_size = args.batch_size
    max_steps = args.max_steps
    num_players = 1
    # train_data = load_data(args.train_path, batch_size=args.batch_size)
    # valid_data = load_data(args.valid_path, batch_size=args.batch_size)

    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'train_{}'.format(args.log_suffix)), flush_secs=1)  
    
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300
    pystk.init(config)
    race_config = pystk.RaceConfig(num_kart=1, track='icy_soccer_field', mode=pystk.RaceConfig.RaceMode.SOCCER)
    race_config.players.pop()
    # for i in range(num_players):
    # o = Player(HockeyPlayer(1)) #TODO: change the code to use the agent to make deecisions but also update with eaech epoch
    # race_config.players.append(o.config)
    i=0
    o = pystk.PlayerConfig(controller = pystk.PlayerConfig.Controller.AI_CONTROL, team = int((i+1)%2.0))
    race_config.players.append(o)

    global_step = 0
    for e in range(args.epochs):
        all_targets = []
        all_predictions = []

        k = pystk.Race(race_config)
        k.start()
        try:
            k.step() #Take on step to start
            state = pystk.WorldState()

            for t in range(max_steps//batch_size):
                batch_targets = []
                batch_images = []
                for b in range(batch_size):
                    print('\rEpoch: {} Step: {} of {}'.format(e,t,max_steps//batch_size), end='\r')
                    state.update()

                    s = k.step()
                    # t_actions = []
                    # for i in range(num_players):
                    i=0
                    la = k.last_action[i]
                    img = torch.tensor(np.array(k.render_data[i].image), dtype=torch.float).to(device).permute(2,0,1)
                    
                    # p = model(img[None])
                    a = torch.tensor((la.steer, la.acceleration, la.brake), dtype=torch.float)
                    batch_images.append(img[None])
                    batch_targets.append(a[None])
                x = torch.cat(batch_images)
                # print(x.shape)
                p = model(x)
                a = torch.cat(batch_targets).to(device)

                optimizer.zero_grad()
                l = loss(p, a)
                l.backward()
                optimizer.step()

                all_targets.append(a.numpy())
                all_predictions.append(p.squeeze().detach().cpu().numpy())
                # print(all_predictions)
                train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
                global_step += 1
        finally:
            k.stop()
            del k
        train_logger.add_scalar('RMSE_steer', getRMSE(all_predictions, all_targets, 0),global_step=e)
        train_logger.add_scalar('RMSE_acceleration', getRMSE(all_predictions, all_targets, 1),global_step=e)
        train_logger.add_scalar('RMSE_brake', getRMSE(all_predictions, all_targets, 2),global_step=e)
        save_model(model, 'temp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train(args)