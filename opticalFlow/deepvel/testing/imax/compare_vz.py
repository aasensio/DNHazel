import numpy as np
import matplotlib.pyplot as pl
import scipy.io as io
import h5py

if (__name__ == '__main__'):

    f = h5py.File('imax_velocity_noPmodes_vz.h5')
    net = f['velocity'][:] * 10.0

    res = io.readsav('velI.idl')
    spec = res['velI'][42:,100:800,100:800]

    f.close()

    f, ax = pl.subplots(nrows=2)

    ax[0].imshow(np.clip(net[0,:,:,0], -4, 4))
    ax[1].imshow(np.clip(spec[0,:,:], -4, 4))
