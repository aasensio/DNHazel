import numpy as np
import matplotlib.pyplot as pl
import json
# import seaborn as sns

f = open('cnns/resnet2_loss.json', 'r')
out = f.read()
dat = json.loads(out)

n = len(dat)

loss = np.zeros((2,n))

for i in range(n):
    loss[0,i] = dat[i]['loss']
    loss[1,i] = dat[i]['val_loss']

pl.semilogy(loss[0,:], label='training')
pl.semilogy(loss[1,:], label='validation')

pl.legend()
pl.show()