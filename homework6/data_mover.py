import os
import re
import sys

DRIVE1 = 'complete set'
DRIVE2 = 'total_combined 2'
if __name__ == '__main__':
    # for t in track:
    #     ids = [re.search('\d+',x).group() for x in os.listdir(os.path.join('drive_data',DRIVE1)) if bool(re.search(t,x)) and bool(re.search('.png',x))]
    files = os.listdir(os.path.join('drive_data',DRIVE1))
    for f in files:
        temp = f[:-4] + '_third' + f[-4:]
        os.rename('drive_data/{}/{}'.format(DRIVE1,f), 'drive_data/{}/{}'.format(DRIVE2,temp))
        print(temp)