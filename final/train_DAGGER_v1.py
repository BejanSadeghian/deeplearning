import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np
import pystk
from agent_dagger.player import HockeyPlayer
from agent_dagger.model import Action, save_model
from agent_dagger.vision_model import Vision, load_vision_model
import itertools
from torch.distributions.normal import Normal
from torchvision import transforms

def getRMSE(list_preds, list_targets, idx):
    predicted = np.array([x[idx] for x in list_preds])
    targets = np.array([x[idx] for x in list_targets])
    return np.sqrt(((predicted - targets)**2).mean())


def rollout_agent(device, vision, action, n_steps=1000):
    race_config = pystk.RaceConfig(num_kart=1, track='icy_soccer_field', mode=pystk.RaceConfig.RaceMode.SOCCER)
    image_to_tensor = transforms.ToTensor()
    k = pystk.Race(race_config)
    k.start()
    for i in range(5): #Skip the first 5 steps since its the game starting
        k.step() 
    try:
        data = []
        for n in range(n_steps):
            # img = torch.tensor(np.array(k.render_data[0].image), dtype=torch.float).to(device).permute(2,0,1)
            last_image = np.array(k.render_data[0].image)
            img = image_to_tensor(last_image)
            heatmap, resized_image = vision(img.to(device))
            # p = action(torch.sigmoid(heatmap))[0]
            combined_image = torch.cat((resized_image, heatmap), 1)
            p = action(combined_image)[0]
            # print(p[0])
            k.step(pystk.Action(steer=float(p[0]), acceleration=float(p[1]), brake=float(p[2])>0.5)) #TODO: remove /10 later
            # print(pystk.Action(acceleration=float(p[0]), steer=float(p[1]), brake=float(p[2])>0.5))
            la = k.last_action[0]
            # print((la.acceleration, la.steer, la.brake))
            # print('end')
            data.append((last_image, (la.steer, la.acceleration, la.brake))) #TODO: remove /10 later
    finally:
        k.stop()
        del k
    return data

def rollout(device, action, n_steps=1000):
    race_config = pystk.RaceConfig(num_kart=1, track='icy_soccer_field', mode=pystk.RaceConfig.RaceMode.SOCCER)
    o = pystk.PlayerConfig(controller = pystk.PlayerConfig.Controller.AI_CONTROL, team = 0)
    race_config.players.append(o)
    k = pystk.Race(race_config)
    
    k.start()
    for i in range(5): #Skip the first 5 steps since its the game starting
        k.step() 
    try:
        data = []
        for n in range(n_steps):
            last_image = np.array(k.render_data[0].image)
            k.step()
            la = k.last_action[0]

            data.append((last_image, (la.steer, la.acceleration, la.brake))) #TODO: remove /10 later 3 in this script and 1 in player
    finally:
        k.stop()
        del k
    return data

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)
    model = Action(normalize=True, inference=False).to(device)
    model.train(True)
    vision_model = load_vision_model('vision') #Vision().to(device)
    vision_model.to(device)
    vision_model.train(False)
    vision_model.eval()

    image_to_tensor = transforms.ToTensor()

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    batch_size = args.batch_size
    max_steps = args.max_steps
    num_players = 1

    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'train_{}'.format(args.log_suffix)), flush_secs=1)  
    
    #PySTK init
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300
    pystk.init(config)

    train_data = list(itertools.chain(*[rollout(device, model) for it in range(10)]))
    global_step = 0
    for e in range(args.epochs):
        print('Epoch:',e)
        all_targets = []
        all_predictions = []

        # state = pystk.WorldState()
        # print('a')

        if e > 1:
            train_data.extend(rollout_agent(device, vision_model, model))
        
        np.random.shuffle(train_data)
        # print(train_data)
        for iteration in range(0, len(train_data)-batch_size+1, batch_size):
            # print('\rEpoch: {} Step: {} of {}'.format(e,iteration//batch_size,batch_size), end='\r')

            batch_data = torch.cat([image_to_tensor(train_data[i][0]).unsqueeze(0) for i in range(iteration, iteration+batch_size)],0)
            batch_label = torch.as_tensor([train_data[i][1] for i in range(iteration, iteration+batch_size)]).float()

            heatmap, resized_image = vision_model(batch_data.to(device))

            combined_data = torch.cat((resized_image, torch.sigmoid(heatmap)),1)

            p = model(combined_data)
            # p = model(batch_data.to(device))


            l = loss(p.cpu(), batch_label.cpu())

            # l = (log_probs.cpu() * torch.abs(samples.cpu() - batch_label.cpu())).sum()
            # print(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            all_targets.append(batch_label.detach().cpu().numpy())
            all_predictions.append(p.squeeze().detach().cpu().numpy())
            if args.logdir is not None:
                train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
            global_step += 1

        # if args.logdir is not None:
        #     train_logger.add_scalar('RMSE_steer', getRMSE(all_predictions, all_targets, 0),global_step=e)
        #     train_logger.add_scalar('RMSE_acceleration', getRMSE(all_predictions, all_targets, 1),global_step=e)
        #     train_logger.add_scalar('RMSE_brake', getRMSE(all_predictions, all_targets, 2),global_step=e)
        save_model(model, 'action')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()
    train(args)