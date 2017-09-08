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

    def apply_psf(self, image):
        image_fft = pyfftw.interfaces.numpy_fft.fft2(image)
        self.image = np.real(pyfftw.interfaces.numpy_fft.ifft2(self.psf_fft * image_fft))
        return self.image

    def rebin_image(self, nx, ny):
        return congrid.resample(self.image, (nx, ny))


def run_master(file_int, file_vv, db_images, db_vel, zero):

    print(" * MASTER : reading {0}...".format(file_int))
    im = io.readsav('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/{0}.save'.format(file_int))['int']
    print(" * MASTER : reading {0}...".format(file_vv))
    vel = io.readsav('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/{0}.save'.format(file_vv))

    n_timesteps, nx_orig, ny_orig = im.shape

    tasks = [i for i in range(n_timesteps)]

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
                dataToSend = {'index': task_index+zero, 'image': im[task_index,0:1008,0:1008], 'vx1': vel['vx1'][task_index,0:1008,0:1008], 
                    'vz1': vel['vz1'][task_index,0:1008,0:1008],
                    'vy1': vel['vy1'][task_index,0:1008,0:1008],
                    'vx01': vel['vx01'][task_index,0:1008,0:1008], 
                    'vz01': vel['vz01'][task_index,0:1008,0:1008], 
                    'vy01': vel['vy01'][task_index,0:1008,0:1008], 
                    'vx001': vel['vx001'][task_index,0:1008,0:1008], 
                    'vz001': vel['vz001'][task_index,0:1008,0:1008],
                    'vy001': vel['vy001'][task_index,0:1008,0:1008]}
                comm.send(dataToSend, dest=source, tag=tags.START)
                print(" * MASTER : sending task {0} to worker {1}".format(task_index, source), flush=True)
                task_index += 1
            else:
                print("Sending termination")
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            index = dataReceived['index']
            im_r = dataReceived['image']
            vx1_r = dataReceived['vx1']
            vz1_r = dataReceived['vz1']
            vy1_r = dataReceived['vy1']
            vx01_r = dataReceived['vx01']
            vz01_r = dataReceived['vz01']
            vy01_r = dataReceived['vy01']
            vx001_r = dataReceived['vx001']
            vz001_r = dataReceived['vz001']
            vy001_r = dataReceived['vy001']
                        
            db_images[index,:,:] = im_r
            db_vel[0,index,:,:] = vx1_r
            db_vel[1,index,:,:] = vz1_r
            db_vel[2,index,:,:] = vx01_r
            db_vel[3,index,:,:] = vz01_r
            db_vel[4,index,:,:] = vx001_r
            db_vel[5,index,:,:] = vz001_r
            db_vel[6,index,:,:] = vy1_r
            db_vel[7,index,:,:] = vy01_r
            db_vel[8,index,:,:] = vy001_r
    
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
            vx1 = dataReceived['vx1']
            vz1 = dataReceived['vz1']
            vy1 = dataReceived['vy1']
            vx01 = dataReceived['vx01']
            vz01 = dataReceived['vz01']
            vy01 = dataReceived['vy01']
            vx001 = dataReceived['vx001']
            vz001 = dataReceived['vz001']
            vy001 = dataReceived['vy001']

            telescope_radius = 0.5 * 0.965 * u.meter
            pixel_size = 0.0545 * u.arcsec / u.pixel
            fov = 1008 * u.pixel
            lambda0 = 500 * u.nm
            imax = imax_degradation(telescope_radius, pixel_size, fov)
            imax.compute_psf(lambda0)

            im = congrid.resample(imax.apply_psf(im), (830, 830))
            vx1 = congrid.resample(imax.apply_psf(vx1), (830, 830))
            vz1 = congrid.resample(imax.apply_psf(vz1), (830, 830))
            vy1 = congrid.resample(imax.apply_psf(vy1), (830, 830))
            vx01 = congrid.resample(imax.apply_psf(vx01), (830, 830))
            vz01 = congrid.resample(imax.apply_psf(vz01), (830, 830))
            vy01 = congrid.resample(imax.apply_psf(vy01), (830, 830))
            vx001 = congrid.resample(imax.apply_psf(vx001), (830, 830))
            vz001 = congrid.resample(imax.apply_psf(vz001), (830, 830))
            vy001 = congrid.resample(imax.apply_psf(vy001), (830, 830))

            dataToSend = {'index': task_index, 'image': im, 'vx1': vx1, 'vz1': vz1, 'vy1': vy1, 'vx01': vx01, 'vz01': vz01, 'vy01': vy01, 'vx001': vx001, 'vz001': vz001, 'vy001': vy001}
            comm.send(dataToSend, dest=0, tag=tags.DONE)            
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)


# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

files_int = ['int_48h1_956', 'int_48h1_1520']#, 'int_48h1_2020q']
files_vel = ['vv_48h1_956', 'vv_48h1_1520']#, 'vv_48h1_2020q']

if rank == 0: 

    fi = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/int_imax_degraded.h5', 'w')
    fv = h5py.File('/net/duna/scratch1/aasensio/deepLearning/opticalFlow/database/v_imax_degraded.h5', 'w')

    db_images = fi.create_dataset("image", (82+53, 830, 830), dtype='float32')
    db_vel = fv.create_dataset("vel", (9, 82+53, 830, 830), dtype='float32')
        
    # Master process executes code below
    n_done = 0
    for i in range(len(files_int)):
        n_done = run_master(files_int[i], files_vel[i], db_images, db_vel, n_done)
        print('n_done', n_done)
      
    fi.close()
    fv.close()
    
else:
    # Worker processes execute code below
    for i in range(len(files_int)):
        run_slave()