import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
import astropy.constants as const
import scipy.interpolate
import scipy.special as sp
import scipy.misc as mi
import struct
from tqdm import tqdm
from ipdb import set_trace as stop
import matplotlib.animation as manimation

class synth_gaussian(object):
    """
    """
    def __init__(self, NSIDE, npix, lmax=10, clv=True):
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
        self.lmax = lmax
        self.n_coef_max = (1+lmax)**2

        # self.rot_velocity = rot_velocity
        self.clv = clv

# Generate the indices of all healpix pixels
        self.indices = np.arange(hp.nside2npix(NSIDE), dtype='int')
        self.n_healpix_pxl = len(self.indices)

# Define the orthographic projector that generates the maps of the star on the plane of the sky
        self.projector = hp.projector.OrthographicProj(xsize=int(self.npix))

# This function returns the pixel associated with a vector (x,y,z). This is needed by the projector
        self.f_vec2pix = lambda x, y, z: hp.pixelfunc.vec2pix(int(self.NSIDE), x, y, z)

        self.polar_angle, self.azimuthal_angle = hp.pixelfunc.pix2ang(self.NSIDE, np.arange(self.n_healpix_pxl))

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
        
# Generate a fake temperature map in the star using spherical harmonics
        # self.temperature_map = 5000 * np.ones(self.npix)
        # self.temperature_map = 5000 + 250 * hp.sphtfunc.alm2map(np.ones(10,dtype='complex'),self.NSIDE) #np.random.rand(self.n_healpix_pxl) * 2000 + 5000 #

        self.temperature_map = 5000 * np.ones(self.hp_npix)
        self.coeffs = hp.sphtfunc.map2alm(self.temperature_map)

        self.nlambda = 500
        self.v = np.linspace(-100.0, 100.0, self.nlambda)
        self.d = 0.5
        self.delta = 5.0

        self.Plm = np.zeros((self.lmax+2, self.lmax+2, self.n_healpix_pxl))
        self.dPlm = np.zeros((self.lmax+2, self.lmax+2, self.n_healpix_pxl))

        self.Clm = np.zeros((self.lmax+2, self.lmax+2))

        for i in range(self.n_healpix_pxl):
            self.Plm[:,:,i], self.dPlm[:,:,i] = sp.lpmn(self.lmax+1, self.lmax+1, np.cos(self.polar_angle[i]))
            self.dPlm[:,:,i] *= -np.sin(self.polar_angle[i])

        for l in range(self.lmax+2):
            for m in range(0, l+1):
                self.Clm[m,l] = np.sqrt((2.0*l+1) / (4.0*np.pi) * mi.factorial(l-m) / mi.factorial(l+m))

        self.lambda0 = 5000.0
        self.lande = 1.2
        self.constant = -4.6686e-13 * self.lambda0 * self.lande * 3e5

    def Km(self, m):
        if (m < 0):
            return np.sin(np.abs(m) * self.azimuthal_angle)
        else:
            return np.cos(np.abs(m) * self.azimuthal_angle)

    def spherical_harmonics(self, m_signed, l):
        m = np.abs(m_signed)
        Y = -self.Clm[m,l] * self.Plm[m,l,:] * self.Km(m)
        Z = self.Clm[m,l] / (l+1.0) * self.dPlm[m,l,:] * self.Km(m_signed)
        X = -self.Clm[m,l] / (l+1.0) * self.Plm[m,l,:] / np.sin(self.polar_angle) * m_signed * self.Km(-m_signed)

        return Y, X, Z

    def get_alpha(self, m, l):
        return self.alpha[(1+(l-1))**2+m]

    def get_beta(self, m, l):
        return self.beta[(1+(l-1))**2+m]

    def get_gamma(self, m, l):
        return self.gamma[(1+(l-1))**2+m]

    def compute_magnetic_field(self, alpha, beta, gamma):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.B_spherical = np.zeros((3,self.n_healpix_pxl))
        self.B_cartesian = np.zeros((3,self.n_healpix_pxl))

        for l in range(self.lmax):
            for m in range(-l, l, 1):
                Y, X, Z = self.spherical_harmonics(m, l)
                self.B_spherical[0,:] -= self.get_alpha(m,l) * Y
                self.B_spherical[1,:] -= self.get_beta(m,l) * Z + self.get_gamma(m,l) * X
                self.B_spherical[2,:] -= self.get_beta(m,l) * Z - self.get_gamma(m,l) * X

                self.B_cartesian[0,:] = np.sin(self.polar_angle) * np.cos(self.azimuthal_angle) * self.B_spherical[0,:] + \
                    np.cos(self.polar_angle) * np.cos(self.azimuthal_angle) * self.B_spherical[1,:] - \
                    np.sin(self.azimuthal_angle) * self.B_spherical[2,:]

                self.B_cartesian[1,:] = np.sin(self.polar_angle) * np.sin(self.azimuthal_angle) * self.B_spherical[0,:] +  \
                    np.cos(self.polar_angle) * np.sin(self.azimuthal_angle) * self.B_spherical[1,:] + \
                    np.cos(self.azimuthal_angle) * self.B_spherical[2,:]

                self.B_cartesian[2,:] = np.cos(self.polar_angle) * self.B_spherical[0,:] - \
                    np.sin(self.polar_angle) * self.B_spherical[1,:]

        
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
        self.rotations = rotations

        self.avg_mu = [None] * self.n_phases
        self.avg_v = [None] * self.n_phases
        self.velocity = [None] * self.n_phases
        self.n_pixel_unique = [None] * self.n_phases
        self.n_pixels = [None] * self.n_phases
        self.pixel_unique = [None] * self.n_phases
        self.rotator = [None] * self.n_phases

        for loop in range(self.n_phases):
            self.rotator[loop] = hp.rotator.Rotator(rotations[loop,:])

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
                self.velocity[loop][i] = self.avg_v[loop][i]


    def compute_stellar_spectrum(self, alpha, beta, gamma, modulus, rot_velocity=0.0):
        """
        Compute the averaged spectrum on the star for a given temperature map and for a given rotation
        Args:
            temperatures (float) : temperatures in all healpix pixels in Kelvin
            rot_velocity (float) : rotation velocity
        
        Returns:
            spectrum (float) : average spectrum
        """
        stokesi = [np.zeros(self.nlambda)] * self.n_phases
        stokesv = [np.zeros(self.nlambda)] * self.n_phases

        self.compute_magnetic_field(alpha, beta, gamma)
        
        for loop in range(self.n_phases):
            total_stokes = 0

            self.B_rotated = np.array(self.rotator[0](self.B_cartesian, inv=True))
            for i in range(self.n_pixel_unique[loop]):
                if (self.avg_mu[loop][i] != 0):
                    out = 1.0 - self.d * np.exp(-(self.v - rot_velocity * self.velocity[loop][i])**2 / self.delta**2)
                    stokesi[loop] = stokesi[loop] + out * self.n_pixels[loop][i]

                    out = 2.0 * self.d * (self.v - rot_velocity * self.velocity[loop][i]) / self.delta**2 * \
                        (np.exp(-(self.v - rot_velocity * self.velocity[loop][i])**2 / self.delta**2))

                    out *= self.constant * self.B_cartesian[0,self.pixel_unique[loop][i]] * modulus

                    stokesv[loop] = stokesv[loop] + out * self.n_pixels[loop][i]

                    total_stokes += self.n_pixels[loop][i]

            stokesi[loop] /= total_stokes
            stokesv[loop] /= total_stokes
            
        return np.array([item for sublist in stokesi for item in sublist]), np.array([item for sublist in stokesv for item in sublist])


    def plot_star(self, stokesv):        
        
        f, ax = pl.subplots(nrows=1, ncols=4, figsize=(18,10))

        stokesv2 = stokesv.reshape((self.n_phases,self.nlambda))

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(codec='libx264', fps=4, bitrate=20000, metadata=metadata, extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(f, "movie.mp4", self.n_phases):
            for loop in tqdm(range(self.n_phases)):

                self.B_rotated = np.array(self.rotator[loop](self.B_cartesian, inv=True))

                for i in range(3):
                    tmp = self.projector.projmap(self.B_rotated[i,:], self.f_vec2pix, rot=self.rotations[loop,:])[:,0:int(self.npix/2)]
                    ax[i].imshow(tmp)

                ax[3].plot(stokesv2[loop,:])

                writer.grab_frame()
                for i in range(4):
                    ax[i].cla()



if (__name__ == '__main__'):
    NSIDE = 8
    npix = 400
    star = synth_gaussian(NSIDE, npix, lmax=10, clv=True)
    noise = 1e-10
    n_phases = 4
    
    angles = np.zeros((n_phases,2))
    for i in range(n_phases):
        angles[i,:] = np.array([360 / n_phases * i,0])

    print("Precomputing rotation maps")
    star.precompute_rotation_maps(angles)

    alpha = np.zeros(star.n_coef_max)
    # alpha[1] = 1.0
    alpha[2] = 0.5
    beta = np.zeros(star.n_coef_max)
    gamma = np.zeros(star.n_coef_max)
    modulus = 1e3

    print("Computing Stokes maps")
    stokesi, stokesv = star.compute_stellar_spectrum(alpha, beta, gamma, modulus, rot_velocity=70.0)
    
    f, ax = pl.subplots(figsize=(12,6))

    ax.plot(star.v, stokesv[0:star.nlambda])

    pl.show()

    # star.plot_star(stokesv)