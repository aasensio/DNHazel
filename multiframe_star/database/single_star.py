import numpy as np
import poppy
import pyfftw
import scipy.special as sp
from astropy import units as u
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

class telescope_degradation(object):

    def __init__(self, telescope_radius, pixel_size, fov, secondary_radius=None):
        self.telescope_radius = telescope_radius
        if (secondary_radius != None):
            self.secondary_radius = secondary_radius
        else:
            self.secondary_radius = 0.0
        self.pixel_size = pixel_size
        self.fov = fov
        self.zernike_max = 45        
        self.nx = fov[0]
        self.ny = fov[1]

    def compute_psf(self, lambda0, r0):
        self.lambda0 = lambda0
        self.r0 = r0

# In-focus PSF
        osys = poppy.OpticalSystem()

        osys.add_pupil(poppy.CircularAperture(radius = self.telescope_radius))
        
        if (self.secondary_radius != 0):
            osys.add_pupil(poppy.SecondaryObscuration(secondary_radius = self.secondary_radius))

# Telescope alone
        osys.add_detector(pixelscale=self.pixel_size, fov_pixels=self.fov, oversample=1)
        psf = osys.calc_psf(wavelength=self.lambda0)

        self.psf = psf[0].data
        nx_psf, ny_psf = psf[0].data.shape

        psf = np.roll(self.psf, int(nx_psf/2), axis=0)
        psf = np.roll(psf, int(ny_psf/2), axis=1)

        self.psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf)
        
# Atmosphere
        phase_atmosphere = _zernike_coeff_kolmogorov(2.0 * self.telescope_radius.to(u.cm).value, self.r0.to(u.cm).value, self.zernike_max)
        phase_atmosphere[0:4] = 0.0

        thinlens = poppy.ZernikeWFE(radius=self.telescope_radius.to(u.m).value, coefficients=phase_atmosphere * self.lambda0.to(u.m).value / (2.0 * np.pi))
        osys.add_pupil(thinlens)

        osys.add_detector(pixelscale=self.pixel_size, fov_pixels=self.fov, oversample=1)

        psf = osys.calc_psf(wavelength=self.lambda0)

        self.psf_atm = psf[0].data
        nx_psf, ny_psf = psf[0].data.shape

        psf = np.roll(self.psf_atm, int(nx_psf/2), axis=0)
        psf = np.roll(psf, int(ny_psf/2), axis=1)

        self.psf_atm_fft = pyfftw.interfaces.numpy_fft.fft2(psf)

    def apply_psf(self, image):
        image_fft = pyfftw.interfaces.numpy_fft.fft2(image)
        self.image = np.real(pyfftw.interfaces.numpy_fft.ifft2(self.psf_fft * image_fft))        
        self.image_atm = np.real(pyfftw.interfaces.numpy_fft.ifft2(self.psf_atm_fft * image_fft))

        return self.image, self.image_atm

def run_master(db_images, n_images, n_done):
        
    tasks = [i for i in range(n_images)]

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
                
                dataToSend = {'index': task_index}
                comm.send(dataToSend, dest=source, tag=tags.START)
                print(" * MASTER : sending task {0} to worker {1}".format(task_index, source), flush=True)
                task_index += 1
            else:
                print("Sending termination")
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            index = dataReceived['index']
            im_r = dataReceived['image']
            im_atm_r = dataReceived['image_atm']
            
            db_images[index,:,:,0] = im_r
            db_images[index,:,:,1] = im_atm_r
                
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
            
            telescope_radius = 0.5 * 1.5 * u.meter
            pixel_size = 0.042 * u.arcsec / u.pixel
            fov = (100, 100) * u.pixel
            lambda0 = 800 * u.nm
            r0 = 10.0 * u.cm
            border = 30
            nx = int(fov.value[0])
            ny = int(fov.value[1])
            imax = telescope_degradation(telescope_radius, pixel_size, (nx,ny))
            imax.compute_psf(lambda0, r0)

            star_field = np.zeros((nx,ny))
            indx = np.random.randint(border, nx-border)
            indy = np.random.randint(border, ny-border)
            star_field[indx, indy] = 1.0

            im, im_atm = imax.apply_psf(star_field)
            
            dataToSend = {'index': task_index, 'image': im, 'image_atm': im_atm}
            comm.send(dataToSend, dest=0, tag=tags.DONE)            
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)


# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object


# if rank == 0: 

#     fi = h5py.File('/net/duna/scratch1/aasensio/deepLearning/stars/database/database.h5', 'w')
#     n_images = 30000
#     db_images = fi.create_dataset("image", (n_images, 100, 100, 2), dtype='float32')
        
#     # Master process executes code below    
#     n_done = run_master(db_images, n_images, 0)
      
#     fi.close()
    
# else:
#     # Worker processes execute code below
#     run_slave()

if rank == 0: 

    fi = h5py.File('/net/duna/scratch1/aasensio/deepLearning/stars/database/database_validation.h5', 'w')
    n_images = 1000
    db_images = fi.create_dataset("image", (n_images, 100, 100, 2), dtype='float32')
        
    # Master process executes code below    
    n_done = run_master(db_images, n_images, 0)
      
    fi.close()
    
else:
    # Worker processes execute code below
    run_slave()