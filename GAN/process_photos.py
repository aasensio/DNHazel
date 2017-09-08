import numpy as np
import matplotlib.pyplot as pl
import glob
from scipy.misc import imread, imresize
from tqdm import tqdm
import h5py

def get_image(image_path, w=64, h=64):
    im = imread(image_path, mode='RGB').astype(np.float)
    orig_h, orig_w = im.shape[:2]
    im = imresize(im, (64, 64))
    return im

w, h = 64, 64

dirs = glob.glob('Photos/*')

names = []

for d in dirs:
    dat = glob.glob('{0}/*.jp*g'.format(d), recursive=True)
    names.append(dat)

names = [item for sublist in names for item in sublist]

data = np.zeros((len(names), w*h*3), dtype=np.uint8)

for n, fname in tqdm(enumerate(names)):
    image = get_image(fname, w, h)
    data[n] = image.flatten()    

f = h5py.File('BEGAN/photos.h5', 'w')
f.create_dataset("images", data=data)
f.close()