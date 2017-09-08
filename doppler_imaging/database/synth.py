import autograd.numpy as np
from autograd import value_and_grad
from autograd.optimizers import adam
# import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
import astropy.constants as const
import scipy.interpolate
import struct
from tqdm import tqdm
from ipdb import set_trace as stop

def soft_thr(x, lambdaPar, lower=None, upper=None):
    out = np.sign(x) * np.fmax(np.abs(x) - lambdaPar, 0)
    if (lower != None):
        out[out < lower] = 0.0
    if (upper != None):
        out[out > upper] = 0.0
    return out

def hard_thr(x, lambdaPar, lower=None, upper=None):
    out = np.copy(x)
    out[np.abs(x) < lambdaPar] = 0.0

    if (lower != None):
        out[out < lower] = 0.0
    if (upper != None):
        out[out > upper] = 0.0
    return out

def _read_kurucz_spec(f):
    """
    Read Kurucz spectra that have been precomputed

    Args:
        f (string) : path to the file to be read
        
    Returns:
        new_vel (real array) : velocity axis in km/s
        spectrum (real array) : spectrum for each velocity bin
    """
    f = open(f, "rb")
    res = f.read()
    
    n_chunk = struct.unpack('i',res[0:4])
    
    freq = []
    stokes = []
    cont = []
    
    left = 4
    
    for i in range(n_chunk[0]):
        
        right = left + 4
        n = struct.unpack('i',res[left:right])

        left = right
        right = left + 4
        nmus = struct.unpack('i',res[left:right])


        left = right
        right = left + 8*n[0]
        t1 = np.asarray(struct.unpack('d'*n[0],res[left:right]))
        freq.append(t1)        
                
        left = right
        right = left + 8*n[0]*nmus[0]

        t2 = np.asarray(struct.unpack('d'*n[0]*nmus[0],res[left:right])).reshape((n[0],nmus[0]))
        stokes.append(t2)

        left = right
        right = left + 8*n[0]*nmus[0]

        t2 = np.asarray(struct.unpack('d'*n[0]*nmus[0],res[left:right])).reshape((n[0],nmus[0]))
        cont.append(t2)
        
        left = right
        
    freq = np.concatenate(freq)
    stokes = np.concatenate(stokes)
    cont = np.concatenate(cont)

    ind = np.argsort(freq)
    freq = freq[ind]
    stokes = stokes[ind]
    cont = cont[ind]
    wavelength = const.c.to('cm/s').value / freq
    mean_wavelength = np.mean(wavelength)

    vel = (wavelength - mean_wavelength) / mean_wavelength * const.c.to('km/s').value

    nl, nmus = stokes.shape

# Reinterpolate in a equidistant velocity axis
    new_vel = np.linspace(np.min(vel), np.max(vel), nl)
    for i in range(nmus):
        interpolator = scipy.interpolate.interp1d(vel, stokes[:,i], kind='linear')
        stokes[:,i] = interpolator(new_vel)

    return new_vel, wavelength, stokes

class synth(object):
    """
    """
    def __init__(self, NSIDE, npix, clv=True):
        """
        Args:
            NSIDE (int) : the healpix NSIDE parameter, must be a power of 2, less than 2**30
            npix (int) : number of pixel in the X and Y axis of the final projected map
            rot_velocity (float) : rotation velocity of the star in the equator in km/s
        
        Returns:
            None
        """
        self.NSIDE = int(NSIDE)
        self.npix = int(npix)
        self.hp_npix = hp.nside2npix(NSIDE)

        # self.rot_velocity = rot_velocity
        self.clv = clv

# Generate the indices of all healpix pixels
        self.indices = np.arange(hp.nside2npix(NSIDE), dtype='int')
        self.n_healpix_pxl = len(self.indices)

# Define the orthographic projector that generates the maps of the star on the plane of the sky
        self.projector = hp.projector.OrthographicProj(xsize=int(self.npix))

# This function returns the pixel associated with a vector (x,y,z). This is needed by the projector
        self.f_vec2pix = lambda x, y, z: hp.pixelfunc.vec2pix(int(self.NSIDE), x, y, z)

# Generate a mesh grid of X and Y points in the plane of the sky that covers only the observed hemisphere of the star
        x = np.linspace(-2.0,0.0,int(self.npix/2))
        y = np.linspace(-1.0,1.0,int(self.npix/2))
        X, Y = np.meshgrid(x,y)

# Rotational velocity vector (pointing in the z direction and unit vector)
        omega = np.array([0,0,1])

# Compute the radial vector at each position in the map and the projected velocity on the plane of the sky
        radial_vector = np.array(self.projector.xy2vec(X.flatten(), Y.flatten())).reshape((3,int(self.npix/2),int(self.npix/2)))        
        self.vel_projection = np.cross(omega[:,None,None], radial_vector, axisa=0, axisb=0)[:,:,0]

# Compute the mu angle (astrocentric angle)
        self.mu_angle = radial_vector[0,:,:]        
        
# Read all Kurucz models from the database. Hardwired temperature and mu angles
        print("Reading Kurucz spectra...")
        self.T = 3500 + 250 * np.arange(27)
        self.mus = np.array([1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.02])[::-1]

        for i in tqdm(range(27)):
            f = 'kurucz_models/RESULTS/T_{0:d}_logg4.0_feh0.0.spec'.format(self.T[i])
            vel, _, spec = _read_kurucz_spec(f)

            if (i == 0):
                self.nlambda, self.nmus = spec.shape
                self.velocity = np.zeros((self.nlambda))
                self.spectrum = np.zeros((27,self.nmus,self.nlambda))

            self.velocity = vel
            self.spectrum[i,:,:] = spec[:,::-1].T

# Generate a fake temperature map in the star using spherical harmonics
        # self.temperature_map = 5000 * np.ones(self.npix)
        # self.temperature_map = 5000 + 250 * hp.sphtfunc.alm2map(np.ones(10,dtype='complex'),self.NSIDE) #np.random.rand(self.n_healpix_pxl) * 2000 + 5000 #

        self.temperature_map = 5000 * np.ones(self.hp_npix)
        self.coeffs = hp.sphtfunc.map2alm(self.temperature_map)

        self.velocity_per_pxl = self.velocity[1] - self.velocity[0]

        self.freq_grid = np.fft.fftfreq(self.nlambda)

        self.gradient = value_and_grad(self.loss)

    def generate_random_star(self):
        """
        Generate a random star (not yet fully implemented)
        Args:
            None
        
        Returns:
            None
        """

        theta = np.linspace(0, np.pi, num=40)[:, None]
        phi = np.linspace(-np.pi, np.pi, num=40)

        pix = hp.ang2pix(self.NSIDE, theta, phi)
        healpix_map = np.zeros(hp.nside2npix(self.NSIDE), dtype=np.double)

        healpix_map[pix] = np.random.randn(40,40)

        stop()


    def compute_rotated_map(self, rotation):
        """
        Compute stellar maps projected on the plane of the sky for a given rotation of the star
        Args:
            rotation (float) : rotation around the star in degrees given as [longitude, latitude] in degrees
        
        Returns:
            pixel_unique (int) : vector with the "active" healpix pixels
            pixel_map (int) : map showing the healpix pixel projected on the plane of the sky
            mu_pixel (float): map of the astrocentric angle for each pixel on the plane of the sky (zero for pixels not in the star)
            T_pixel (float): map of temperatures for each pixel on the plane of the sky
        """
        mu_pixel = np.zeros_like(self.mu_angle)
        T_pixel = np.zeros_like(self.mu_angle)

# Get the projection of the healpix pixel indices on the plane of the sky
        pixel_map = self.projector.projmap(self.indices, self.f_vec2pix, rot=rotation)[:,0:int(self.npix/2)]

# Get the unique elements in the vector
        pixel_unique = np.unique(pixel_map)
        
# Now loop over all unique pixels, filling up the array of the projected map with the mu and temeperature values
        for j in range(len(pixel_unique)):
            ind = np.where(pixel_map == pixel_unique[j])            

            if (np.all(np.isfinite(self.mu_angle[ind[0],ind[1]]))):
                if (self.mu_angle[ind[0],ind[1]].size == 0):
                    value = 0.0
                else:                    
                    value = np.nanmean(self.mu_angle[ind[0],ind[1]])
                    mu_pixel[ind[0],ind[1]] = value

                    T_pixel[ind[0],ind[1]] = self.temperature_map[int(pixel_unique[j])]
            else:
                mu_pixel[ind[0],ind[1]] = 0.0
                T_pixel[ind[0],ind[1]] = 0.0

        return pixel_unique, pixel_map, mu_pixel, T_pixel

    def show_rotated_map(self):        
        """
        Little utility to show rotated maps
        Args:
            None
        
        Returns:
            None
        """
        f, ax = pl.subplots(nrows=4, ncols=6, figsize=(12,10))
        ax = ax.flatten()
        for i in range(24):            
            pixel_unique, pixel_map, pixel_mu, pixel_T = self.compute_rotated_map([15*i,0])
            im = ax[i].imshow(pixel_T, cmap=pl.cm.viridis)
            pl.colorbar(im, ax=ax[i])
        # im = ax[1].imshow(pixel_mu, cmap=pl.cm.viridis)
        # pl.colorbar(im, ax=ax[1])
        pl.tight_layout()
        pl.show()

    def shift_spectrum(self, spec, shift_in_pxl):
        kernel = np.exp(-1j*2*np.pi*self.freq_grid*shift_in_pxl)

        return np.real(np.fft.ifft( np.fft.fft(spec) * kernel ))

    def get_spectra(self, T, mu, velocity):
        """
        Return the Kurucz spectrum for a value of the temperature, astrocentric angle and velocity. It linearly
        interpolates on the set of computed spectra
        Args:
            T (float) : temperature in K
            mu (float) : astrocentric angle in the range [0,1]
            velocity (float) : velocity shift of the spectrum in km/s
        
        Returns:
            spectrum (float) : interpolated and shifted spectrum
        """
        idT = np.searchsorted(self.T, T) - 1
        idmu = np.searchsorted(self.mus, mu) - 1

# Simple bilinear interpolation
        xd = (T - self.T[idT]) / (self.T[idT+1] - self.T[idT])
        yd = (mu - self.mus[idmu]) / (self.mus[idmu+1] - self.mus[idmu])

        c0 = self.spectrum[idT,idmu,:] * (1.0 - xd) + self.spectrum[idT+1,idmu,:] * xd
        c1 = self.spectrum[idT,idmu+1,:] * (1.0 - xd) + self.spectrum[idT+1,idmu+1,:] * xd

        tmp = np.squeeze(c0 * (1.0 - yd) + c1 * yd)

# Velocity shift in pixel units
        shift_in_pxl = velocity / self.velocity_per_pxl

        return self.shift_spectrum(tmp, shift_in_pxl) #nd.shift(tmp, shift_in_pxl)

    def precompute_rotation_maps(self, rotations=None):
        """
        Compute the averaged spectrum on the star for a given temperature map and for a given rotation
        Args:
            rotations (float) : [N_phases x 2] giving [longitude, latitude] in degrees for each phase
        
        Returns:
            None
        """
        if (rotations is None):
            print("Use some angles for the rotations")
            return

        self.n_phases = rotations.shape[0]

        self.avg_mu = [None] * self.n_phases
        self.avg_v = [None] * self.n_phases
        self.velocity = [None] * self.n_phases
        self.n_pixel_unique = [None] * self.n_phases
        self.n_pixels = [None] * self.n_phases
        self.pixel_unique = [None] * self.n_phases

        for loop in range(self.n_phases):
            mu_pixel = np.zeros_like(self.mu_angle)
            v_pixel = np.zeros_like(self.vel_projection)
        
            pixel_map = self.projector.projmap(self.indices, self.f_vec2pix, rot=rotations[loop,:])[:,0:int(self.npix/2)]
            pixel_unique = np.unique(pixel_map[np.isfinite(pixel_map)])

            for j in range(len(pixel_unique)):
                ind = np.where(pixel_map == pixel_unique[j])

                if (np.all(np.isfinite(self.mu_angle[ind[0],ind[1]]))):
                    if (self.mu_angle[ind[0],ind[1]].size == 0):
                        mu_pixel[ind[0],ind[1]] = 0.0
                        v_pixel[ind[0],ind[1]] = 0.0
                    else:                    
                        
                        if (self.clv):
                            value = np.nanmean(self.mu_angle[ind[0],ind[1]])
                        else:
                            value = 1.0

                        mu_pixel[ind[0],ind[1]] = value

                        value = np.nanmean(self.vel_projection[ind[0],ind[1]])
                        v_pixel[ind[0],ind[1]] = value
                else:
                    mu_pixel[ind[0],ind[1]] = 0.0
                    v_pixel[ind[0],ind[1]] = 0.0

            self.n_pixel_unique[loop] = len(pixel_unique)
            self.avg_mu[loop] = np.zeros(self.n_pixel_unique[loop])
            self.avg_v[loop] = np.zeros(self.n_pixel_unique[loop])
            self.velocity[loop] = np.zeros(self.n_pixel_unique[loop])
            self.n_pixels[loop] = np.zeros(self.n_pixel_unique[loop], dtype='int')
            self.pixel_unique[loop] = pixel_unique.astype('int')

            for i in range(len(pixel_unique)):
                ind = np.where(pixel_map == pixel_unique[i])
                self.n_pixels[loop][i] = len(ind[0])
                self.avg_mu[loop][i] = np.unique(mu_pixel[ind[0], ind[1]])
                self.avg_v[loop][i] = np.unique(v_pixel[ind[0], ind[1]])            
                self.velocity[loop][i] = self.avg_mu[loop][i] * self.avg_v[loop][i]


    def compute_stellar_spectrum(self, temperatures, rot_velocity=0.0):
        """
        Compute the averaged spectrum on the star for a given temperature map and for a given rotation
        Args:
            temperatures (float) : temperatures in all healpix pixels in Kelvin
            rot_velocity (float) : rotation velocity
        
        Returns:
            spectrum (float) : average spectrum
        """
        spectrum = [np.zeros(self.nlambda)] * self.n_phases

        for loop in range(self.n_phases):
            total = 0
            for i in range(self.n_pixel_unique[loop]):
                if (self.avg_mu[loop][i] != 0):
                    out = self.get_spectra(temperatures[self.pixel_unique[loop][i]], self.avg_mu[loop][i], rot_velocity * self.velocity[loop][i])
                    spectrum[loop] = spectrum[loop] + out * self.n_pixels[loop][i]
                    total = total + self.n_pixels[loop][i]                

            spectrum[loop] /= total
            spectrum[loop] = spectrum[loop] / spectrum[loop][150]

        return np.array([item for sublist in spectrum for item in sublist])

    def loss(self, temperatures):
        return np.sum((star.compute_stellar_spectrum(temperatures, rot_velocity=20.0) - self.obs)**2 / self.noise**2) / (self.n_phases * self.nlambda)

    def optimize(self, noise, threshold=0.0):

        self.noise = noise

        h = 5e6
        temperature_map = 6000 * np.ones(self.hp_npix)
        for i in range(6):
            chi2, grad = star.gradient(temperature_map)
            change = -h * grad
            temperature_map = temperature_map + change

            if (threshold != 0):
                coeffs = hp.sphtfunc.map2alm(temperature_map)
                real_part = hard_thr(coeffs.real, threshold)
                imag_part = hard_thr(coeffs.imag, threshold)
                coeffs = real_part + 1j*imag_part
                temperature_map = hp.sphtfunc.alm2map(coeffs, self.NSIDE)

            print('It: {0} - chi2={1}'.format(i, chi2))

        return temperature_map, hp.sphtfunc.map2alm(temperature_map)

    def optimize_adam(self, noise, num_iters=20, step_size=0.001, b1=0.9, b2=0.999, eps=1e-8):

        self.noise = noise

        x = 6000 * np.ones(self.hp_npix)

        m = np.zeros(len(x))
        v = np.zeros(len(x))
        for i in range(num_iters):
            chi2, grad = star.gradient(x)
            
            m = (1 - b1) * grad + b1 * m  # First  moment estimate.
            v = (1 - b2) * (grad**2) + b2 * v  # Second moment estimate.
            mhat = m / (1 - b1**(i + 1))    # Bias correction.
            vhat = v / (1 - b2**(i + 1))
            x = x - step_size*mhat/(np.sqrt(vhat) + eps)

            print('It: {0} - chi2={1}'.format(i, chi2))

        return x

if (__name__ == '__main__'):
    NSIDE = 8
    npix = 400
    star = synth(NSIDE, npix, clv=True)
    noise = 1e-2
    
    angles = np.zeros((4,2))
    for i in range(4):
        angles[i,:] = np.array([90*i,0])
    star.precompute_rotation_maps(angles)

    spectrum_obs = star.compute_stellar_spectrum(star.temperature_map, rot_velocity=20.0)

    spectrum_obs += np.random.normal(loc=0.0, scale=noise, size=spectrum_obs.shape)
    star.obs = spectrum_obs

    # final = star.optimize_adam(noise, step_size=2e5)
    final, coeffs = star.optimize(noise, threshold=1e0)

    spectrum = star.compute_stellar_spectrum(final, rot_velocity=20.0)

    f, ax = pl.subplots(figsize=(12,6))

    ax.plot(spectrum_obs)
    ax.plot(spectrum)

    pl.show()



