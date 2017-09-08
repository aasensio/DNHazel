import numpy as np
import h5py
import scipy.io as io
from ipdb import set_trace as stop

def update_moments(xa, xb, M2a, M2b, na, nb):
    delta = xb - xa
    xx = (na*xa + nb*xb) / (na+nb)
    M2x = M2a + M2b + delta**2 * na * nb / (na+nb)

    return xx, M2x

meanI = np.zeros(4)
stdI = np.zeros(4)

labels = ['int_48h1_415', 'int_48h1_956', 'int_48h1_1520', 'int_48h1_2020q']

for i in range(4):
    print("Reading {0}...".format(labels[i]))
    im = io.readsav('/scratch1/3dcubes/stein/{0}.save'.format(labels[i]))['int']
    meanI[i] = np.mean(im)
    stdI[i] = np.std(im)

im = 0.0

minx = np.zeros((3,4))
miny = np.zeros((3,4))
maxx = np.zeros((3,4))
maxy = np.zeros((3,4))

labels = ['vv_48h1_415', 'vv_48h1_956', 'vv_48h1_1520', 'vv_48h1_2020q']

for i in range(4):
    print("Reading {0}...".format(labels[i]))
    vel = io.readsav('/scratch1/3dcubes/stein/{0}.save'.format(labels[i]))

    minx[0,i] = np.min(vel['vx1'])
    miny[0,i] = np.min(vel['vz1'])
    maxx[0,i] = np.max(vel['vx1'])
    maxy[0,i] = np.max(vel['vz1'])

    minx[1,i] = np.min(vel['vx01'])
    miny[1,i] = np.min(vel['vz01'])
    maxx[1,i] = np.max(vel['vx01'])
    maxy[1,i] = np.max(vel['vz01'])

    minx[2,i] = np.min(vel['vx001'])
    miny[2,i] = np.min(vel['vz001'])
    maxx[2,i] = np.max(vel['vx001'])
    maxy[2,i] = np.max(vel['vz001'])

np.savez('normalizations.npz', meanI, stdI, minx, miny, maxx, maxy)