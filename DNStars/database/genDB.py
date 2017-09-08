import numpy as np
import h5py
import matplotlib.pyplot as pl
import pyiacsun as ps
from ipdb import set_trace as stop


labels = ['metals', 'C', 'N', 'O', 'alpha', 'log10vdop', 'Teff', 'logg']
dims = np.asarray([7, 9, 5, 9, 5, 11, 11])
npix = np.asarray([2920, 2400, 1893])
lower = np.asarray([-2.5, -1.0, -1.0, -1.0, -0.301029995663981, 3500.0, 0.0])
step = np.asarray([0.5, 0.25, 0.5, 0.25, 0.301029995663981, 250.0, 0.5])
wave0 = np.asarray([4.18093204498, 4.20088815689, 4.21747207642])
wave_step = 6.00000021223e-6

axis = []
for i in range(7):
    axis.append(lower[i] + step[i] * np.arange(dims[i]))

n_total_models = np.prod(dims)
n_lambda = np.sum(npix)
n_pars = len(dims)

# logg, Teff, log10vdop, OMgSiSCaTi, metals

f = open('/scratch/aasensio/deepLearning/DNStars/database/f_apsKK-01-10k_w123.dat', 'r')
for i in range(98):
    res = f.readline()

f5 = h5py.File('/scratch/aasensio/deepLearning/DNStars/database/database.h5', 'a')
databasePars = f5.create_dataset("parameters", (n_total_models, n_pars), dtype='float32')
databaseFlux = f5.create_dataset("flux", (n_total_models, n_lambda), dtype='float32')

ind = np.random.permutation(n_total_models)
loop = 0

ps.util.progressbar(0, n_total_models)

for i_metals in range(dims[0]):
    for i_C in range(dims[1]):
        for i_N in range(dims[2]):
            for i_alpha in range(dims[3]):
                for i_vdop in range(dims[4]):
                    for i_teff in range(dims[5]):
                        for i_logg in range(dims[6]):
                            res = f.readline()
                            res = np.asarray(res[0:-1].split('   '), dtype='float32')
                            databaseFlux[ind[loop],:] = res

                            databasePars[ind[loop],:] = [axis[0][i_metals], axis[1][i_C], axis[2][i_N], axis[3][i_alpha], axis[4][i_vdop], axis[5][i_teff], axis[6][i_logg]]

                            loop += 1

                            if (loop % 100 == 0):
                                ps.util.progressbar(loop, n_total_models)

f5.close()

