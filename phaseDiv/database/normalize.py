import numpy as np
import h5py

f = h5py.File('/scratch1/3dcubes/cheung/images_imax_degraded.h5', 'r')
im = f.get("image")

min_i = np.min(im[:,1,0:200,0:200])
max_i = np.max(im[:,1,0:200,0:200])
median_i = np.median(im[:,1,0:200,0:200])
f.close()

np.savez('/scratch/Dropbox/GIT/DeepLearning/phaseDiv/database/normalization.npz', min_i, max_i, median_i)