import numpy as np
import h5py
import glob
from tqdm import tqdm
from ipdb import set_trace as stop

def process(models, profiles):
    delta_models = models[1:,:,:] - models[0:-1,:,:]
    delta_models[:,1,:] /= 1000.0
    delta_profiles = profiles[:,0,:,:] - profiles[:,1,:,:]

    return delta_models, delta_profiles

def write_file(files, output):
    n = len(files)
    f = h5py.File(output, 'w')
    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    database_models_shape = f.create_dataset("models_shape", (2,), dtype='int')
    database_profiles_shape = f.create_dataset("profiles_shape", (3,), dtype='int')
    database_stokes = f.create_dataset("stokes", (n,), dtype=dt)
    database_models = f.create_dataset("models", (n,), dtype=dt)
    database_iterations = f.create_dataset("niterations", (n,), dtype='int')

    for i in tqdm(range(n)):
        dat = np.load(files[i], encoding='latin1').item()
        models = dat['modelos']
        profiles = dat['perfiles']
        niteration = models.shape[0]

        delta_models, delta_profiles = process(models, profiles)

        stop()

        if (i == 0):
            database_models_shape = models.shape[1:]
            database_profiles_shape = profiles.shape[1:]

        database_stokes[i] = profiles.flatten()
        database_models[i] = models.flatten()
        database_iterations[i] = niteration

    f.close()

# files = glob.glob('/net/izar/scratch/carlos/INVERSOR/TRAINING/*.npy')
files = glob.glob('carlos/*.npy')
n_files = len(files)

write_file(files[0:n_files-10], 'training_nicole.h5')
write_file(files[n_files-10:], 'validation_nicole.h5')
