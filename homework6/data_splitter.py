#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:31:20 2019

@author: bsadeghian
"""

import re
import os
import sys
import numpy as np

if __name__ == '__main__':
    pct = 0.8
    track = ['zengarden','lighthouse','hacienda','snowtuxpeak','cornfield_crossing','scotland']
    for t in track:
        ids = [re.search('\d+',x).group() for x in os.listdir('drive_data') if bool(re.search(t,x)) and bool(re.search('.png',x))]
        train_size = int(len(ids) * pct)
        train_ids = np.random.choice(ids, size=train_size, replace=False)
        valid_ids = list(set(ids) - set(train_ids))
#        os.mkdir('drive_data/train')
#        os.mkdir('drive_data/valid')
        for tr in train_ids:
            for ext in ['.csv', '.png']:
                os.rename('drive_data/{}_{}{}'.format(t,tr,ext), 'drive_data/train/{}_{}{}'.format(t,tr,ext))
        for tr in valid_ids:
            for ext in ['.csv', '.png']:
                os.rename('drive_data/{}_{}{}'.format(t,tr,ext), 'drive_data/valid/{}_{}{}'.format(t,tr,ext))
            
