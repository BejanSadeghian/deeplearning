from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data

def train(args):
    model = model_factory[args.model]()
   
    train_gen = load_data(args.train_path, batch_size=args.batch_size)
    valid_gen = load_data(args.validate_path, batch_size=args.batch_size)
    
    loss = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum)
    
    for iteration in range(args.epochs):
        print('iteration',iteration)
        counter = 0
        for data_batch in train_gen:           
            p_y = model(data_batch[0])
            actual = data_batch[1]
            
            ## Update weights using the optimizer calculcated gradients
            optimizer.zero_grad()
            l = loss(p_y, actual)
            l.backward()
            optimizer.step()    
            print('epoch: {}, train batch: {}; loss: {}; accuracy: {}'.format(iteration, counter, l.item(), accuracy(p_y, actual)))
            counter += 1

    save_model(model)
    
    #Validate
    counter = 0
    for data_batch in valid_gen:   
        counter += 1
        
        p_y = model(data_batch[0])
        actual = data_batch[1]
        
        l = loss(p_y, actual)
        print('validation batch: {}; loss: {}; accuracy: {}'.format(counter, l.item(), accuracy(p_y, actual)))
        counter += 1

    ## Validate
#    data_loader = load_data(args.train_path, batch_size=args.batch_size)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-t', '--train_path', type=str)
    parser.add_argument('-v', '--validate_path', type=str)
    parser.add_argument('-ep', '--epochs', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-mom', '--momentum', type=float, default=0.9)

    args = parser.parse_args()
    train(args)
