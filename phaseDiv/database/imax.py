import numpy as np
import poppy
import pyfftw
import scipy.special as sp
from astropy import units as u
import scipy.io as io
import congrid
import h5py
from mpi4py import MPI
from enum import IntEnum

class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3


def _even(x):
    return x%2 == 0

def _zernike_parity(j, jp):
    return _even(j-jp)

def _zernike_coeff_kolmogorov(D, r0, n_zernike):
    """
    Return Zernike coefficients in phase units
    """
    covariance = np.zeros((n_zernike,n_zernike))
    for j in range(n_zernike):
        n, m = poppy.zernike.noll_indices(j+1)
        for jpr in range(n_zernike):
            npr, mpr = poppy.zernike.noll_indices(jpr+1)
            
            deltaz = (m == mpr) and (_zernike_parity(j, jpr) or m == 0)
            
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

class imax_degradation(object):

    def __init__(self, telescope_radius, pixel_size, fov, secondary_radius=None):
        self.telescope_radius = telescope_radius
        if (secondary_radius != None):
            self.secondary_radius = secondary_radius
        else:
            self.secondary_radius = 0.0
        self.pixel_size = pixel_size
        self.fov = fov
        self.zernike_max = 45
        self.r0 = 10.0 * u.cm

    def compute_psf(self, lambda0, rms_telescope=1.0/9.0, rms_atmosphere=1.0/9.0):
        self.lambda0 = lambda0

# In-focus PSF
        osys = poppy.OpticalSystem()

        osys.add_pupil(poppy.CircularAperture(radius = self.telescope_radius))
        
        if (self.secondary_radius != 0):
            osys.add_pupil(poppy.SecondaryObscuration(secondary_radius = self.secondary_radius))

# Telescope diffraction + aberrations
        phase_telescope = np.random.randn(self.zernike_max)
        sigma = np.sqrt(np.sum(phase_telescope[4:]**2)) / 2.0 / np.pi
        phase_telescope *= rms_telescope / sigma
        phase_telescope[0:4] = 0.0
        
# Atmosphere
        phase_atmosphere = _zernike_coeff_kolmogorov(2.0 * self.telescope_radius.to(u.cm).value, self.r0.to(u.cm).value, self.zernike_max)
        sigma = np.sqrt(np.sum(phase_atmosphere[4:]**2)) / 2.0 / np.pi
        phase_atmosphere *= rms_atmosphere / sigma
        phase_atmosphere[0:4] = 0.0

        thinlens = poppy.ZernikeWFE(radius=self.telescope_radius.to(u.m).value, coefficients=(phase_telescope + phase_atmosphere) * lambda0.to(u.m).value / (2.0 * np.pi))
        osys.add_pupil(thinlens)

        osys.add_detector(pixelscale=self.pixel_size, fov_pixels=self.fov, oversample=1)

        psf = osys.calc_psf(wavelength=self.lambda0)        

        self.psf = psf[0].data
        nx_psf, ny_psf = psf[0].data.shape

        psf = np.roll(self.psf, int(nx_psf/2), axis=0)
        psf = np.roll(psf, int(ny_psf/2), axis=1)

        self.psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf)

# Out-of-focus PSF
        n_waves_defocus = 1.0
        defocus_coefficient = n_waves_defocus * lambda0.to(u.m).value / (2.0 * np.sqrt(3))        
        osys = poppy.OpticalSystem()

        osys.add_pupil(poppy.CircularAperture(radius = self.telescope_radius))
        
        if (self.secondary_radius != 0):
            osys.add_pupil(poppy.SecondaryObscuration(secondary_radius = self.secondary_radius))

        zernike = (phase_telescope + phase_atmosphere) * lambda0.to(u.m).value / (2.0 * np.pi)
        zernike[3] += defocus_coefficient

        thinlens = poppy.ZernikeWFE(radius=self.telescope_radius.to(u.m).value, coefficients=zernike)
        osys.add_pupil(thinlens)

        osys.add_detector(pixelscale=self.pixel_size, fov_pixels=self.fov, oversample=1)

        psf_defocus = osys.calc_psf(wavelength=self.lambda0)        

        self.psf_defocus = psf_defocus[0].data
        nx_psf, ny_psf = psf_defocus[0].data.shape

        psf_defocus = np.roll(self.psf_defocus, int(nx_psf/2), axis=0)
        psf_defocus = np.roll(psf_defocus, int(ny_psf/2), axis=1)

        self.psf_defocus_fft = pyfftw.interfaces.numpy_fft.fft2(psf_defocus)

    def apply_psf(self, image):
        image_fft = pyfftw.interfaces.numpy_fft.fft2(image)
        self.image = np.real(pyfftw.interfaces.numpy_fft.ifft2(self.psf_fft * image_fft))        
        self.image_defocus = np.real(pyfftw.interfaces.numpy_fft.ifft2(self.psf_defocus_fft * image_fft))

        return self.image, self.image_defocus

    def rebin_image(self, nx, ny):
        return congrid.resample(self.image, (nx, ny))

def run_master(db_images, n_done):
        
    tasks = [i for i in range(250)]

    task_index = 0
    num_workers = size - 1
    closed_workers = 0
    print("*** Master starting with {0} workers".format(num_workers))
    while closed_workers < num_workers:
        dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)                
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
            # Worker is ready, so send it a task
            if task_index < len(tasks):

                im = np.memmap('/net/viga/scratch1/3dcubes/cheung/I_out.{0:06d}'.format(1000*task_index),dtype='float32',offset=4*4,mode='r',shape=(1024,1920))

                dataToSend = {'index': task_index, 'image': im}
                comm.send(dataToSend, dest=source, tag=tags.START)
                print(" * MASTER : sending task {0} to worker {1}".format(task_index, source), flush=True)
                task_index += 1
            else:
                print("Sending termination")
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            index = dataReceived['index']
            im_r = dataReceived['image']
            im_focus_r = dataReceived['image_focus']
            im_defocus_r = dataReceived['image_defocus']
            
            db_images[index,0,:,:] = im_r
            db_images[index,1,:,:] = im_focus_r
            db_images[index,2,:,:] = im_defocus_r
                
            print(" * MASTER : got block {0} from worker {1}".format(index, source), flush=True)
            
        elif tag == tags.EXIT:
            print(" * MASTER : worker {0} exited.".format(source))
            closed_workers += 1

    print("Master block finished")
    return len(tasks)

def run_slave():    
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
        
        if tag == tags.START:            
            # Do the work here
            task_index = dataReceived['index']
            im = dataReceived['image']

            telescope_radius = 0.5 * 0.965 * u.meter
            pixel_size = 0.0545 * u.arcsec / u.pixel
            fov = (1024, 1920) * u.pixel            
            lambda0 = 500 * u.nm
            imax = imax_degradation(telescope_radius, pixel_size, fov)
            imax.compute_psf(lambda0)

            im_focus, im_defocus = imax.apply_psf(im)

            im = congrid.resample(im, (843, 1580))
            im_focus = congrid.resample(im_focus, (843, 1580))
            im_defocus = congrid.resample(im_defocus, (843, 1580))

            dataToSend = {'index': task_index, 'image': im, 'image_focus': im_focus, 'image_defocus': im_defocus}
            comm.send(dataToSend, dest=0, tag=tags.DONE)            
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)


# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

if rank == 0: 

    fi = h5py.File('/net/viga/scratch1/3dcubes/cheung/images_imax_degraded.h5', 'w')

    db_images = fi.create_dataset("image", (250, 3, 843, 1580), dtype='float32')
        
    # Master process executes code below    
    n_done = run_master(db_images, 0)
      
    fi.close()
    
else:
    # Worker processes execute code below
    run_slave()