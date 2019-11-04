#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:45:02 2019

@author: bsadeghian
"""

from homework.train import train
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir')
# Put custom arguments here
parser.add_argument('-t', '--train_path', type=str)
parser.add_argument('-v', '--valid_path', type=str)
parser.add_argument('-ep', '--epochs', type=int, default=500)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
parser.add_argument('-mom', '--momentum', type=float, default=0.9)
parser.add_argument('-ls', '--log_suffix', type=str, default='')
parser.add_argument('-cl','--clear_cache', type=bool, default=True)
parser.add_argument('-rot', '--data_rotate', type=bool, default=False)
parser.add_argument('-flip', '--data_flip', type=bool, default=False)
parser.add_argument('-jit', '--data_colorjitter', type=bool, default=False)
parser.add_argument('-la', '--layers', type=str)
parser.add_argument('-ml', '--model_label', type=str)

if __name__ == '__main__':
    import itertools

    hyper_params = {
        '--layers':[[32,32,64], [32,32,64,128], [32,32,64,64,128], [32,32,64,64,128,128]],
        '--learning_rate':['0.01', '1e-3'],
        # '--data_rotate':['False', 'True'],
        # '--data_flip':['False', 'True'],
        # '--data_colorjitter':['False', 'True']
        }

    static = {
        '--train_path':'drive_data/train',
        '--valid_path':'drive_data/valid',
        '--epochs':'100'
        }

    L = []
    order = []
    for h in hyper_params.keys():
        order.append(h)
        L.append(hyper_params[h])
    
    for s in static.keys():
        order.append(s)
        L.append([static[s]])
    product = list(itertools.product(*L))

    for hyper_opt in product:
        arguments = []
        arguments.append('--log_dir')
        arguments.append('log')
        arguments.append('--batch_size')
        arguments.append('32')
        arguments.append('--model_label')
        arguments.append('Run'+str(hyper_opt))
        arguments.append('--log_suffix')
        arguments.append('Run'+str(hyper_opt))
        for i, label in enumerate(order):
            arguments.append(label)
            if label == '--layers':
                # for x in hyper_opt[i]:
                arguments.append(str(hyper_opt[i]))
            else:
                arguments.append(hyper_opt[i])
        args = parser.parse_args(arguments)
        print(args)
        try:
            train(args)
        except Exception as e:
            print(e,args)
    
