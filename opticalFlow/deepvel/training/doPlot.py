import numpy as np
import matplotlib.pyplot as pl
import pickle
import h5py

f = h5py.File('/scratch/aasensio/deepLearning/opticalFlow/database/database_velocity_validation.h5', 'r')
vel = f.get("velocity")[0:6,:,:,:]

f = h5py.File('/scratch/aasensio/deepLearning/opticalFlow/database/database_images_validation.h5', 'r')
inten = f.get("intensity")[0:6,:,:,:]

with open('cnns/test2_pred.pkl', 'rb') as f:
    out = pickle.load(f)

pl.close('all')
f, ax = pl.subplots(ncols=4, nrows=5, figsize=(15,15))

for i in range(5):
    ax[i,0].imshow(out[i,:,:,0])
    ax[i,1].imshow(vel[i,:,:,0])
    ax[i,2].imshow(out[i,:,:,1])    
    ax[i,3].imshow(vel[i,:,:,1])

pl.show()