import numpy as np
import h5py
from ipdb import set_trace as stop
from tqdm import tqdm

input_training = "/net/viga/scratch1/deepLearning/doppler_imaging/database/training_stars.h5"
        
f = h5py.File(input_training, 'r')
n_training = len(f['modulus'])
min_alpha = 1e99 * np.ones(16)
max_alpha = -1e99 * np.ones(16)
for i in tqdm(range(n_training)):
    min_alpha = np.min(np.vstack([min_alpha, f['alpha'][i]]).T, axis=1)
    max_alpha = np.max(np.vstack([max_alpha, f['alpha'][i]]).T, axis=1)

normalizations = np.zeros((2,16))
normalizations[0,:] = min_alpha
normalizations[1,:] = max_alpha

np.save('normalization.npy', normalizations)