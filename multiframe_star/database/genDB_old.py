import numpy as np
import h5py
import scipy.io as io
import poppy
import sys
import scipy.special as sp
import pyfftw
from astropy import units as u
import matplotlib.pyplot as pl
from ipdb import set_trace as stop

def even(x):
    return x%2 == 0

def zernike_parity(j, jp):
    return even(j-jp)

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

def zernike_coeff_kolmogorov(D, r0, n_zernike):
    """
    Return Zernike coefficients in phase units
    """
    covariance = np.zeros((n_zernike,n_zernike))
    for j in range(n_zernike):
        n, m = poppy.zernike.noll_indices(j+1)
        for jpr in range(n_zernike):
            npr, mpr = poppy.zernike.noll_indices(jpr+1)
            
            deltaz = (m == mpr) and (zernike_parity(j, jpr) or m == 0)
            
            if (deltaz):                
                phase = (-1.0)**(0.5*(n+npr-2*m))
                t1 = np.sqrt((n+1)*(npr+1)) 
                t2 = sp.gamma(14./3.0) * sp.gamma(11./6.0)**2 * (24.0/5.0*sp.gamma(6.0/5.0))**(5.0/6.0) / (2.0*np.pi**2)

                Kzz = t2 * t1 * phase
                
                t1 = sp.gamma(0.5*(n+npr-5.0/3.0)) * (D / r0)**(5.0/3.0)
                t2 = sp.gamma(0.5*(n-npr+17.0/3.0)) * sp.gamma(0.5*(npr-n+17.0/3.0)) * sp.gamma(0.5*(n+npr+23.0/3.0))
                covariance[j,jpr] = Kzz * t1 / t2


    covariance[0,0] = 1.0

    out = np.random.multivariate_normal(np.zeros(n_zernike), covariance)

    out[0:3] = 0.0

    return out

def zero_pad(image, nPixBorder):
    """
    Pad an image using zero
        
    Args:
        image (real): image to be padded
        nPixBorder (int): number of pixel on the border
        
    Returns:
        real: final image
    """
    return np.pad(image, ((nPixBorder,nPixBorder), (nPixBorder,nPixBorder)), mode='constant', constant_values = ((0,0),(0,0)))

def hanning_window(image, percentage):
    """
    Return a Hanning window in 2D
        
    Args:
        size (int): size of the final image
        percentage (TYPE): percentage of the image that is apodized
        
    Returns:
        real: 2D apodization mask
            
    """     
    nx, ny = image.shape
    M = np.ceil(nx * percentage/100.0)
    win = np.hanning(M)

    winOut = np.ones(nx)

    winOut[0:int(M/2)] = win[0:int(M/2)]
    winOut[-int(M/2):] = win[-int(M/2):]

    mean = np.mean(image)

    return (image - mean)* np.outer(winOut, winOut) + mean


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

##############
# Training set
##############
#     for i in range(n_patches):
#         progressbar(i, n_patches, text='Progress (traininig set)')
#         star_field = np.zeros((nx, ny))
#         star_field[int(nx/2), int(ny/2)] = 1.0

#         # Save original image in file
#         database_images[i,0,:,:,0] = star_field

#         star_field_fft = pyfftw.interfaces.numpy_fft.fft2(star_field)
        
#         for j in range(n_frames):
#             r0 = np.random.uniform(low=5.0, high=20.0) * u.cm
            
# # Generate wavefront and defocused wavefront
#             zernike = lambda0.to(u.m).value / (2.0 * np.pi) * zernike_coeff_kolmogorov(telescope_radius.to(u.cm).value, r0.to(u.cm).value, n_zernike)
        
# # Now save perturbed image
#             osys = poppy.OpticalSystem()
#             osys.add_pupil(poppy.CircularAperture(radius = telescope_radius))
#             osys.add_pupil(poppy.SecondaryObscuration(secondary_radius = secondary_radius))
#             thinlens = poppy.ZernikeWFE(radius=telescope_radius.to(u.m).value, coefficients=zernike)
#             osys.add_pupil(thinlens)
#             osys.add_detector(pixelscale=pixSize, fov_arcsec=fov, oversample=1)
#             psf = osys.calc_psf(wavelength=lambda0)

#             nx_psf, ny_psf = psf[0].data.shape
#             psf_roll = np.roll(psf[0].data, int(nx_psf/2), axis=0)
#             psf_roll = np.roll(psf_roll, int(ny_psf/2), axis=1)
            
#             psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf_roll)

#             image_final = np.real(pyfftw.interfaces.numpy_fft.ifft2(psf_fft * star_field_fft))

#             database_images[i,j+1,:,:,0] = image_final

##############
# Validation set
##############
    for i in range(n_patches_validation):
        progressbar(i, n_patches, text='Progress (traininig set)')
        star_field = np.zeros((nx, ny))
        star_field[int(nx/2), int(ny/2)] = 1.0

        # Save original image in file
        database_images_validation[i,0,:,:,0] = star_field

        star_field_fft = pyfftw.interfaces.numpy_fft.fft2(star_field)
        
        for j in range(n_frames):
            r0 = np.random.uniform(low=5.0, high=20.0) * u.cm
            
# Generate wavefront and defocused wavefront
            zernike = lambda0.to(u.m).value / (2.0 * np.pi) * zernike_coeff_kolmogorov(telescope_radius.to(u.cm).value, r0.to(u.cm).value, n_zernike)
        
# Now save perturbed image
            osys = poppy.OpticalSystem()
            osys.add_pupil(poppy.CircularAperture(radius = telescope_radius))
            osys.add_pupil(poppy.SecondaryObscuration(secondary_radius = secondary_radius))
            thinlens = poppy.ZernikeWFE(radius=telescope_radius.to(u.m).value, coefficients=zernike)
            osys.add_pupil(thinlens)
            osys.add_detector(pixelscale=pixSize, fov_arcsec=fov, oversample=1)
            psf = osys.calc_psf(wavelength=lambda0)

            nx_psf, ny_psf = psf[0].data.shape
            psf_roll = np.roll(psf[0].data, int(nx_psf/2), axis=0)
            psf_roll = np.roll(psf_roll, int(ny_psf/2), axis=1)
            
            psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf_roll)

            image_final = np.real(pyfftw.interfaces.numpy_fft.ifft2(psf_fft * star_field_fft))

            database_images_validation[i,j+1,:,:,0] = image_final
                
    f_images.close()
    f_images_validation.close()

if (__name__ == '__main__'):
        
    generate_training(100, 100, 1, 10)