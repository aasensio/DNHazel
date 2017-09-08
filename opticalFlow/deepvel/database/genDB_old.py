import numpy as np
import h5py
import scipy.io as io
from sklearn.feature_extraction import image
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


def generate_training(files_int, files_vel, n_patches_list, n_patches_validation_list):

    n_files = len(files_int)

    np.random.seed(123)

    tmp = np.load('normalizations.npz')

    meanI, stdI, minx, miny, maxx, maxy = tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3'], tmp['arr_4'], tmp['arr_5']

    meanI = np.mean(meanI[1:])
    stdI = np.mean(stdI[1:])

    minx = np.min(minx[:,1:], axis=1)
    miny = np.min(miny[:,1:], axis=1)

    maxx = np.max(maxx[:,1:], axis=1)
    maxy = np.max(maxy[:,1:], axis=1)

    nx = 156
    ny = 156

    nx_v = 68
    ny_v = 68

    n_patches_total = sum(n_patches_list)
    n_patches_validation_total = sum(n_patches_validation_list)

    f_images = h5py.File('/scratch1/aasensio/deepLearning/opticalFlow/database/database_images.h5', 'w')
    f_velocity = h5py.File('/scratch1/aasensio/deepLearning/opticalFlow/database/database_velocity.h5', 'w')

    f_images_validation = h5py.File('/scratch1/aasensio/deepLearning/opticalFlow/database/database_images_validation.h5', 'w')
    f_velocity_validation = h5py.File('/scratch1/aasensio/deepLearning/opticalFlow/database/database_velocity_validation.h5', 'w')

    database_images = f_images.create_dataset('intensity', (n_patches_total, nx, ny, 3), 'f')
    database_velocity = f_velocity.create_dataset('velocity', (n_patches_total, nx_v, ny_v, 6), 'f')

    database_images_validation = f_images_validation.create_dataset('intensity', (n_patches_validation_total, nx, ny, 3), 'f')
    database_velocity_validation = f_velocity_validation.create_dataset('velocity', (n_patches_validation_total, nx_v, ny_v, 6), 'f')

    loop = 0
    loop_val = 0

    ordering = np.random.permutation(n_patches_total)
    
    for j in range(n_files):

        n_patches = n_patches_list[j]
        n_patches_validation = n_patches_validation_list[j]    

        print("Reading {0}...".format(files_int[j]))
        im = io.readsav('/scratch1/aasensio/deepLearning/opticalFlow/database/{0}.save'.format(files_int[j]))['int']

# Not strictly correct because I should correct by the number of points in each set of data, but... :)
        im -= meanI
        im /= stdI

        n_timesteps, nx_orig, ny_orig = im.shape

        print("Reading {0}...".format(files_vel[j]))
        vel = io.readsav('/scratch1/aasensio/deepLearning/opticalFlow/database/{0}.save'.format(files_vel[j]))

# Normalize all velocities to the range [0,1] and save the values    
        velx_1 = vel['vx1'] - minx[0]
        velx_1 /= (maxx[0] - minx[0])
        vely_1 = vel['vz1'] - miny[0]
        vely_1 /= (maxy[0] - miny[0])

        velx_01 = vel['vx01'] - minx[1]
        velx_01 /= (maxx[1] - minx[1])
        vely_01 = vel['vz01'] - miny[1]
        vely_01 /= (maxy[1] - miny[1])

        velx_001 = vel['vx001'] - minx[2]
        velx_001 /= (maxx[2] - minx[2])
        vely_001 = vel['vz001'] - miny[2]
        vely_001 /= (maxy[2] - miny[2])

        pos_x = np.random.randint(low=0, high=nx_orig-nx, size=n_patches)
        pos_y = np.random.randint(low=0, high=ny_orig-ny, size=n_patches)
        pos_t = np.random.randint(low=0, high=n_timesteps-3, size=n_patches)      

        print("Saving training set...")
        for i in range(n_patches):

            index = ordering[loop]

            progressbar(i, n_patches, text='Progress', end=False)
            database_images[index,:,:,0] = im[pos_t[i], pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]
            database_images[index,:,:,1] = im[pos_t[i]+1, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]
            database_images[index,:,:,2] = im[pos_t[i]+2, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]

            database_velocity[index,:,:,0] = velx_1[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]
            database_velocity[index,:,:,1] = vely_1[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]

            database_velocity[index,:,:,2] = velx_01[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]
            database_velocity[index,:,:,3] = vely_01[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]

            database_velocity[index,:,:,4] = velx_001[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]
            database_velocity[index,:,:,5] = vely_001[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]

            loop += 1

        pos_x = np.random.randint(low=0, high=nx_orig-nx, size=n_patches_validation)
        pos_y = np.random.randint(low=0, high=ny_orig-ny, size=n_patches_validation)
        pos_t = np.random.randint(low=0, high=n_timesteps-3, size=n_patches_validation)

        print("Saving validation set...")
        for i in range(n_patches_validation):
            progressbar(i, n_patches_validation, text='Progress', end=False)
            database_images_validation[loop_val,:,:,0] = im[pos_t[i], pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]
            database_images_validation[loop_val,:,:,1] = im[pos_t[i]+1, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]
            database_images_validation[loop_val,:,:,2] = im[pos_t[i]+2, pos_x[i]:pos_x[i]+nx, pos_y[i]:pos_y[i]+ny]

            database_velocity_validation[loop_val,:,:,0] = velx_1[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]
            database_velocity_validation[loop_val,:,:,1] = vely_1[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]

            database_velocity_validation[loop_val,:,:,2] = velx_01[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]
            database_velocity_validation[loop_val,:,:,3] = vely_01[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]

            database_velocity_validation[loop_val,:,:,4] = velx_001[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]
            database_velocity_validation[loop_val,:,:,5] = vely_001[pos_t[i], pos_x[i]+44:pos_x[i]+44+nx_v, pos_y[i]+44:pos_y[i]+44+ny_v]

            loop_val += 1
        
    f_images.close()
    f_velocity.close()


if (__name__ == '__main__'):
    files_int = ['int_48h1_956', 'int_48h1_1520', 'int_48h1_2020q']
    files_vel = ['vv_48h1_956', 'vv_48h1_1520', 'vv_48h1_2020q']

    n_patches_list = [10000, 10000, 10000]
    n_patches_validation_list = [100, 100, 100]

    generate_training(files_int, files_vel, n_patches_list, n_patches_validation_list)