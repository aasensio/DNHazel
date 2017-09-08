import numpy as np
import matplotlib.pyplot as pl

cont = np.load('continuum.npz')
hmi_cont = cont['arr_0']
target_cont = cont['arr_1']
network_cont = cont['arr_2']

blos = np.load('blos.npz')
hmi_blos = blos['arr_0']
target_blos = blos['arr_1']
network_blos = blos['arr_2']

f, ax = pl.subplots(nrows=3, ncols=3, figsize=(9,8), sharex='col')

ind = [4,1,3]

for i in range(3):
    ax[i,0].imshow(hmi_cont[ind[i],:,:,0])
    ax[i,1].imshow(network_cont[ind[i],:,:,0])
    ax[i,2].imshow(target_cont[ind[i],:,:,0])

ax[0,0].set_title('Synthetic - HMI')
ax[0,1].set_title('Network')
ax[0,2].set_title('Synthetic - Target')
pl.tight_layout()
pl.show()
pl.savefig('validation_cont.pdf')


f, ax = pl.subplots(nrows=3, ncols=3, figsize=(9,8), sharex='col')

ind = [4,1,3]

for i in range(3):
    ax[i,0].imshow(hmi_blos[ind[i],:,:,0])
    ax[i,1].imshow(network_blos[ind[i],:,:,0])
    ax[i,2].imshow(target_blos[ind[i],:,:,0])

pl.tight_layout()
pl.show()
pl.savefig('validation_blos.pdf')