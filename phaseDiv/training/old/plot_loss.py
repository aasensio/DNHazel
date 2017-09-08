import numpy as np
import matplotlib.pyplot as pl
import json
import seaborn as sns

def plot_loss(file, label, color):
    f = open(file, 'r')
    out = f.read()
    dat = json.loads(out)

    n = len(dat)

    loss = np.zeros((2,n))

    for i in range(n):
        loss[0,i] = dat[i]['loss']
        loss[1,i] = dat[i]['val_loss']

    pl.semilogy(loss[0,:], label='training {0}'.format(label), color=color)
    pl.semilogy(loss[1,:], '--', label='validation {0}'.format(label), color=color)

pl.close('all')

cmap = sns.color_palette()
plot_loss('cnns/resnet_loss.json', 'resnet', cmap[0])
plot_loss('cnns/resnet_relu_loss.json', 'resnet_relu', cmap[1])
plot_loss('cnns/resnet_relu_noise2e-4_loss.json', 'resnet_noise', cmap[2])
pl.legend()
pl.show()
