import numpy as np
import h5py
import scipy.io as io
import sys
from ipdb import set_trace as stop

def progressbar(current, total, text=None, width=30, end=False):
    """Progress bar
    
    Args:
        current (float): current value of the bar
        total (float): total of the bar
        text (string): additional text to show
        width (int, optional): number of spaces of the bar
        end (bool, optional): end character
    
    Returns:
        None: None
    """
    bar_width = width
    block = int(round(bar_width * current/total))
    text = "\rProgress {3} : [{0}] {1} of {2}".\
        format("#"*block + "-"*(bar_width-block), current, total, text)
    if end:
        text = text +'\n'
    sys.stdout.write(text)
    sys.stdout.flush()


def generate_training(n_patches, n_patches_validation):

    np.random.seed(123)

    tmp = np.load('/net/vena/scratch/Dropbox/GIT/DeepLearning/phaseDiv/database/normalization.npz')

    _, _, median_i = tmp['arr_0'], tmp['arr_1'], tmp['arr_2']

    nx = 50
    ny = 50
    
    f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_images.h5', 'w')
    f_images_validation = h5py.File('/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_images_validation.h5', 'w')

    database_images = f_images.create_dataset('intensity', (n_patches, nx, ny, 3), 'f')
    database_images_validation = f_images_validation.create_dataset('intensity', (n_patches_validation, nx, ny, 3), 'f')
    
    f_db_im = h5py.File('/net/vena/scratch1/3dcubes/cheung/images_imax_degraded.h5', 'r')
    im = f_db_im.get("image")    
        
    n_timesteps, _, nx_orig, ny_orig = im.shape

    pos_x = np.random.randint(low=0, high=nx_orig-nx, size=n_patches+n_patches_validation)
    pos_y = np.random.randint(low=0, high=ny_orig-ny, size=n_patches+n_patches_validation)
    pos_t = np.random.randint(low=0, high=n_timesteps, size=n_patches+n_patches_validation)

    print("Saving training set...")
    loop = 0

    for i in range(n_patches):

        progressbar(loop, n_patches, text='Progress', end=False)
        database_images[loop,:,:,0] = im[pos_t[i], 0, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny] / median_i
        database_images[loop,:,:,1] = im[pos_t[i], 1, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny] / median_i 
        database_images[loop,:,:,2] = im[pos_t[i], 2, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny] / median_i 
        
        loop += 1
    
    print("Saving validation set...")
    loop_val = 0

    for i in range(n_patches,n_patches+n_patches_validation,1):
        progressbar(loop_val, n_patches_validation, text='Progress', end=False)
        database_images_validation[loop_val,:,:,0] = im[pos_t[i], 0, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny] / median_i 
        database_images_validation[loop_val,:,:,1] = im[pos_t[i], 1, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny] / median_i
        database_images_validation[loop_val,:,:,2] = im[pos_t[i], 2, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny] / median_i 
        
        loop_val += 1
        
    f_images.close()

if (__name__ == '__main__'):
    n_patches = 50000
    n_patches_validation = 3000

    generate_training(n_patches, n_patches_validation)