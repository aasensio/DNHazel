import numpy as np
import h5py
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

    tmp = np.load('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/normalization.npz')

    min_i, max_i, min_v, max_v = tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3']
    
    nx = 50
    ny = 50

    nx_v = 50
    ny_v = 50
    
    f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_images.h5', 'w')
    f_velocity = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_velocity.h5', 'w')

    f_images_validation = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_images_validation.h5', 'w')
    f_velocity_validation = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/database_velocity_validation.h5', 'w')

    database_images = f_images.create_dataset('intensity', (n_patches, nx, ny, 2), 'f')
    database_velocity = f_velocity.create_dataset('velocity', (n_patches, nx_v, ny_v, 9), 'f')

    database_images_validation = f_images_validation.create_dataset('intensity', (n_patches_validation, nx, ny, 2), 'f')
    database_velocity_validation = f_velocity_validation.create_dataset('velocity', (n_patches_validation, nx_v, ny_v, 9), 'f')

    f_db_im = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/int_imax_degraded.h5', 'r')
    im = f_db_im.get("image")
    f_db_vel = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/v_imax_degraded.h5', 'r')
    vel = f_db_vel.get("vel")
        
    n_timesteps, nx_orig, ny_orig = im.shape

    im /= np.median(im)

    vel -= min_v[:,None,None,None]
    vel /= (max_v[:,None,None,None] - min_v[:,None,None,None])

    pos_x = np.random.randint(low=0, high=nx_orig-nx, size=n_patches+n_patches_validation)
    pos_y = np.random.randint(low=0, high=ny_orig-ny, size=n_patches+n_patches_validation)
    pos_t = np.random.randint(low=0, high=n_timesteps-2, size=n_patches+n_patches_validation)

    print("Saving training set...")
    loop = 0

    for i in range(n_patches):

        progressbar(loop, n_patches, text='Progress', end=False)
        database_images[loop,:,:,0] = im[pos_t[i], pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]
        database_images[loop,:,:,1] = im[pos_t[i]+1, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]
        
        database_velocity[loop,:,:,0] = vel[0, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity[loop,:,:,1] = vel[1, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]

        database_velocity[loop,:,:,2] = vel[2, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity[loop,:,:,3] = vel[3, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]

        database_velocity[loop,:,:,4] = vel[4, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity[loop,:,:,5] = vel[5, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]

        database_velocity[loop,:,:,6] = vel[6, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity[loop,:,:,7] = vel[7, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity[loop,:,:,8] = vel[8, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]        

        loop += 1
    
    print("Saving validation set...")
    loop_val = 0

    for i in range(n_patches,n_patches+n_patches_validation,1):
        progressbar(loop_val, n_patches_validation, text='Progress', end=False)
        database_images_validation[loop_val,:,:,0] = im[pos_t[i], pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]
        database_images_validation[loop_val,:,:,1] = im[pos_t[i]+1, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]
        
        database_velocity_validation[loop_val,:,:,0] = vel[0, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity_validation[loop_val,:,:,1] = vel[1, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]

        database_velocity_validation[loop_val,:,:,2] = vel[2, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity_validation[loop_val,:,:,3] = vel[3, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]

        database_velocity_validation[loop_val,:,:,4] = vel[4, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity_validation[loop_val,:,:,5] = vel[5, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]

        database_velocity_validation[loop_val,:,:,6] = vel[6, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity_validation[loop_val,:,:,7] = vel[7, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]
        database_velocity_validation[loop_val,:,:,8] = vel[8, pos_t[i], pos_x[i]:pos_x[i]+nx_v, pos_y[i]:pos_y[i]+ny_v]        

        loop_val += 1
        
    f_images.close()
    f_velocity.close()


if (__name__ == '__main__'):
    n_patches = 30000
    n_patches_validation = 1000

    generate_training(n_patches, n_patches_validation)