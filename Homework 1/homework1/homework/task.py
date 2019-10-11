import argparse
import train


if __name__ == '__main__':

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