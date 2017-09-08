import numpy as np
import h5py
import scipy.io as io
import sys
import scipy.special as sp
import pyfftw
from astropy import units as u
import matplotlib.pyplot as pl
from ipdb import set_trace as stop
from soapy import confParse, SCI, atmosphere

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

def generate_training(n_patches, n_patches_validation, n_stars, n_frames):

# Size of final images
    nx = 256
    ny = 256
    n_zernike = 40

# GREGOR
    telescope_radius = 1.44 * 1.440 * u.meter
    secondary_radius = 0.404 * 0.404 * u.meter
    pixSize = (6.0/512.0) * u.arcsec / u.pixel
    lambda0 = 850.0 * u.nm
    fov = 3.0 * u.arcsec    
    border = 100

    f_images = h5py.File('database.h5', 'w')
    f_images_validation = h5py.File('database_validation.h5', 'w')
    database_images = f_images.create_dataset('intensity', (n_patches, n_frames+1, nx, ny, 1), 'f')    
    database_images_validation = f_images_validation.create_dataset('intensity', (n_patches_validation, n_frames+1, nx, ny, 1), 'f')    

    loop = 0
    loop_val = 0

    # load a sim config that defines lots of science cameras across the field
    config = confParse.loadSoapyConfig('sh_8x8.py')

# Init a science camera
    sci_camera = SCI.PSF(config, mask=np.ones((154,154)))

# init some atmosphere
    atmos = atmosphere.atmos(config)

##############
# Training set
##############
    for i in range(n_patches):
        progressbar(i, n_patches, text='Progress (traininig set)')

        star_field = np.zeros((nx, ny))
        indx = np.random.randint(border, nx-border)
        indy = np.random.randint(border, ny-border)
        star_field[indx, indy] = 1.0

        # Save original image in file
        database_images[i,0,:,:,0] = star_field

        star_field_fft = pyfftw.interfaces.numpy_fft.fft2(star_field)
        
        for j in range(n_frames):

            # Get phase for this time step
            phase_scrns = atmos.moveScrns()

# Calculate all the PSF for this turbulence
            psf = sci_camera.frame(phase_scrns)

            nx_psf, ny_psf = psf.shape
            psf_roll = np.roll(psf.data, int(nx_psf/2), axis=0)
            psf_roll = np.roll(psf_roll, int(ny_psf/2), axis=1)
            
            psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf_roll)

            image_final = np.real(pyfftw.interfaces.numpy_fft.ifft2(psf_fft * star_field_fft))

            database_images[i,j+1,:,:,0] = image_final

        for j in range(50):
            phase_scrns = atmos.moveScrns()

##############
# Validation set
##############
    for i in range(n_patches_validation):
        progressbar(i, n_patches_validation, text='Progress (validation set)')

        star_field = np.zeros((nx, ny))
        indx = np.random.randint(border, nx-border)
        indy = np.random.randint(border, ny-border)
        star_field[indx, indy] = 1.0

        # Save original image in file
        database_images_validation[i,0,:,:,0] = star_field

        star_field_fft = pyfftw.interfaces.numpy_fft.fft2(star_field)
        
        for j in range(n_frames):

            # Get phase for this time step
            phase_scrns = atmos.moveScrns()

# Calculate all the PSF for this turbulence
            psf = sci_camera.frame(phase_scrns)

            nx_psf, ny_psf = psf.shape
            psf_roll = np.roll(psf.data, int(nx_psf/2), axis=0)
            psf_roll = np.roll(psf_roll, int(ny_psf/2), axis=1)
            
            psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf_roll)

            image_final = np.real(pyfftw.interfaces.numpy_fft.ifft2(psf_fft * star_field_fft))

            database_images_validation[i,j+1,:,:,0] = image_final

        for j in range(50):
            phase_scrns = atmos.moveScrns()

    f_images.close()
    f_images_validation.close()

if (__name__ == '__main__'):
        
    generate_training(100, 5, 1, 1)