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

    out[0] = 0.0

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


def generate_training(files_int, n_patches_list, n_patches_validation_list):

    n_files = len(files_int)

    tmp = np.load('normalizations.npz')
    meanI, stdI, minx, miny, maxx, maxy = tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3'], tmp['arr_4'], tmp['arr_5']
    meanI = np.mean(meanI[1:])
    stdI = np.mean(stdI[1:])

# Size of final images
    nx = 128
    ny = 128
    n_zernike = 20

# Size of intermediate images to accommodate the apodization
    nx_intermediate = 160
    ny_intermediate = 160
    
    n_patches_total = sum(n_patches_list)
    n_patches_validation_total = sum(n_patches_validation_list)

# GREGOR
    telescope_radius = 1.44 * 1.440 * u.meter
    secondary_radius = 0.404 * 0.404 * u.meter
    pixSize = 0.066 * u.arcsec / u.pixel
    lambda0 = 500.0 * u.nm
    fov = 3 * u.arcsec    
    n_waves_defocus = 1.0
    defocus_coefficient = n_waves_defocus * lambda0.to(u.m).value / (2.0 * np.sqrt(3))

    f_images = h5py.File('/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database.h5', 'w')
    f_images_validation = h5py.File('/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_validation.h5', 'w')
    database_images = f_images.create_dataset('intensity', (n_patches_total, nx, ny, 3), 'f')    
    database_images_validation = f_images_validation.create_dataset('intensity', (n_patches_validation_total, nx, ny, 3), 'f')    

    f_pars = h5py.File('/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_pars.h5', 'w')
    f_pars_validation = h5py.File('/net/duna/scratch1/aasensio/deepLearning/phaseDiv/database/database_pars_validation.h5', 'w')
    database_zernike = f_pars.create_dataset('zernike', (n_patches_total, n_zernike), 'f')
    database_r0 = f_pars.create_dataset('r0', (n_patches_total, 1), 'f')    
    database_zernike_validation = f_pars_validation.create_dataset('zernike', (n_patches_total, n_zernike), 'f')
    database_r0_validation = f_pars_validation.create_dataset('r0', (n_patches_total, 1), 'f')

    loop = 0
    loop_val = 0

    for i_files in range(n_files):
        print("Working with file {0}.save".format(files_int[i_files]))
        im = io.readsav('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/{0}.save'.format(files_int[i_files]))['int']
        n_timesteps, nx_orig, ny_orig = im.shape

        im -= meanI
        im /= stdI

        n_patches = n_patches_list[i_files]
        n_patches_validation = n_patches_validation_list[i_files]

##############
# Training set
##############
        pos_x = np.random.randint(low=0, high=nx_orig-nx_intermediate, size=n_patches)
        pos_y = np.random.randint(low=0, high=ny_orig-ny_intermediate, size=n_patches)
        pos_t = np.random.randint(low=0, high=n_timesteps-1, size=n_patches)

        for ind in range(n_patches):

            progressbar(ind, n_patches, text='Progress (traininig set)')

            r0 = np.random.uniform(low=5.0, high=50.0) * u.cm
            
# Generate wavefront and defocused wavefront
            zernike = []
            zernike.append(lambda0.to(u.m).value / (2.0 * np.pi) * zernike_coeff_kolmogorov(telescope_radius.to(u.cm).value, r0.to(u.cm).value, n_zernike))
            zernike.append(lambda0.to(u.m).value / (2.0 * np.pi) * zernike_coeff_kolmogorov(telescope_radius.to(u.cm).value, r0.to(u.cm).value, n_zernike))
            zernike[1][3] += defocus_coefficient

            database_zernike[loop,:] = zernike[0]
            database_r0[loop] = r0.to(u.cm).value

# Get subimage and apply Hanning window
            image = im[pos_t[ind], pos_x[ind]:pos_x[ind]+nx_intermediate, pos_y[ind]:pos_y[ind]+ny_intermediate]
            image = hanning_window(image, 10)
            image_fft = pyfftw.interfaces.numpy_fft.fft2(image)

# Save original image in file
            database_images[loop,:,:,0] = image[16:-16,16:-16]
            
# Now save perturbed image and defocused image
            for i in range(2):
                osys = poppy.OpticalSystem()
                
                osys.add_pupil(poppy.CircularAperture(radius = telescope_radius))
                # osys.add_pupil(poppy.SecondaryObscuration(secondary_radius = secondary_radius))
                thinlens = poppy.ZernikeWFE(radius=telescope_radius.to(u.m).value, coefficients=zernike[i])
                osys.add_pupil(thinlens)
                osys.add_detector(pixelscale=pixSize, fov_arcsec=fov)
                psf = osys.calc_psf(wavelength=lambda0)

                nx_psf, ny_psf = psf[0].data.shape
                psf_pad = zero_pad(psf[0].data, int((nx_intermediate - nx_psf) / 2))
                psf_pad = np.roll(psf_pad, int(nx_intermediate/2), axis=0)
                psf_pad = np.roll(psf_pad, int(ny_intermediate/2), axis=1)
                
                psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf_pad)

                image_final = np.real(pyfftw.interfaces.numpy_fft.ifft2(psf_fft * image_fft))
                
                database_images[loop,:,:,1+i] = image_final[16:-16,16:-16]

            loop += 1

##############
# Validation set
##############
        pos_x = np.random.randint(low=0, high=nx_orig-nx_intermediate, size=n_patches_validation)
        pos_y = np.random.randint(low=0, high=ny_orig-ny_intermediate, size=n_patches_validation)
        pos_t = np.random.randint(low=0, high=n_timesteps-1, size=n_patches_validation)

        for ind in range(n_patches_validation):

            progressbar(ind, n_patches, text='Progress (validation set)')

            r0 = 20.0 * np.random.rand() * u.cm
            
# Generate wavefront and defocused wavefront
            zernike = []
            zernike.append(lambda0.to(u.m).value / (2.0 * np.pi) * zernike_coeff_kolmogorov(telescope_radius.to(u.cm).value, r0.to(u.cm).value, n_zernike))
            zernike.append(lambda0.to(u.m).value / (2.0 * np.pi) * zernike_coeff_kolmogorov(telescope_radius.to(u.cm).value, r0.to(u.cm).value, n_zernike))
            zernike[1][3] += defocus_coefficient

            database_zernike_validation[loop_val,:] = zernike[0]
            database_r0_validation[loop_val] = r0.to(u.cm).value

# Get subimage and apply Hanning window
            image = im[pos_t[ind], pos_x[ind]:pos_x[ind]+nx_intermediate, pos_y[ind]:pos_y[ind]+ny_intermediate]
            image = hanning_window(image, 5)
            image_fft = pyfftw.interfaces.numpy_fft.fft2(image)

# Save original image in file
            database_images_validation[loop_val,:,:,0] = image[16:-16,16:-16]
            
# Now save perturbed image and defocused image
            for i in range(2):
                osys = poppy.OpticalSystem()
                
                osys.add_pupil(poppy.CircularAperture(radius = telescope_radius))
                # osys.add_pupil(poppy.SecondaryObscuration(secondary_radius = secondary_radius))
                thinlens = poppy.ZernikeWFE(radius=telescope_radius.value, coefficients=zernike[i])
                osys.add_pupil(thinlens)
                osys.add_detector(pixelscale=pixSize, fov_arcsec=fov)
                psf = osys.calc_psf(lambda0)

                nx_psf, ny_psf = psf[0].data.shape
                psf_pad = zero_pad(psf[0].data, int((nx_intermediate - nx_psf) / 2))
                psf_pad = np.roll(psf_pad, int(nx_intermediate/2), axis=0)
                psf_pad = np.roll(psf_pad, int(ny_intermediate/2), axis=1)
                
                psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf_pad)

                image_final = np.real(pyfftw.interfaces.numpy_fft.ifft2(psf_fft * image_fft))
                
                database_images_validation[loop_val,:,:,1+i] = image_final[16:-16,16:-16]

            loop_val += 1
                
    f_images.close()
    f_images_validation.close()
    f_pars.close()
    f_pars_validation.close()

if (__name__ == '__main__'):
    files_int = ['int_48h1_956', 'int_48h1_1520', 'int_48h1_2020q']
    
    n_patches_list = [10000, 10000, 10000]
    n_patches_validation_list = [100, 100, 100]
    
    generate_training(files_int, n_patches_list, n_patches_validation_list)