import numpy as np
import h5py

f = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/int_imax_degraded.h5', 'r')
im = f.get("image")

min_i = np.min(im)
max_i = np.max(im)
f.close()

f = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/v_imax_degraded.h5', 'r')
vel = f.get("vel")

min_v = np.min(vel, axis=(1,2,3))
max_v = np.max(vel, axis=(1,2,3))
f.close()

np.savez('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/normalization.npz', min_i, max_i, min_v, max_v)