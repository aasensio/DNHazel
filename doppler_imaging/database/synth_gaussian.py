import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
import astropy.constants as const
import scipy.interpolate
import scipy.special as sp
import scipy.misc as mi
import struct
from tqdm import tqdm
import matplotlib.animation as manimation
from matplotlib.widgets import Slider

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

class synth_gaussian(object):
    """
    """
    def __init__(self, NSIDE, lmax=10, clv=True):
        """
        Args:
            NSIDE (int) : the healpix NSIDE parameter, must be a power of 2, less than 2**30
            npix (int) : number of pixel in the X and Y axis of the final projected map
            rot_velocity (float) : rotation velocity of the star in the equator in km/s
        
        Returns:
            None
        """
        self.NSIDE = int(NSIDE)        
        self.hp_npix = hp.nside2npix(NSIDE)
        self.lmax = lmax
        self.n_coef_max = (1+lmax)**2
        self.epsilon = 1e-3
        self.l = np.arange(self.lmax+1)

        self.alpha = np.zeros(self.n_coef_max)
        self.beta = np.zeros(self.n_coef_max)
        self.gamma = np.zeros(self.n_coef_max)

        # self.rot_velocity = rot_velocity
        self.clv = clv

# Generate the indices of all healpix pixels
        self.indices = np.arange(hp.nside2npix(NSIDE), dtype='int')
        self.n_healpix_pxl = len(self.indices)

        self.polar_angle, self.azimuthal_angle = hp.pixelfunc.pix2ang(self.NSIDE, np.arange(self.n_healpix_pxl))
        self.pixel_vectors = np.array(hp.pixelfunc.pix2vec(self.NSIDE, self.indices))

# Compute LOS rotation velocity as v=w x r
        self.rotation_velocity = np.cross(np.array([0.0,0.0,1.0])[:,None], self.pixel_vectors, axisa=0, axisb=0, axisc=0)

        self.vec_boundaries = np.zeros((3,4,self.n_healpix_pxl))
        for i in range(self.n_healpix_pxl):
            self.vec_boundaries[:,:,i] = hp.boundaries(self.NSIDE, i)

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
        return self.alpha[(1+(l-1))**2+l+m]

    def get_beta(self, m, l):
        return self.beta[(1+(l-1))**2+l+m]

    def get_gamma(self, m, l):
        return self.gamma[(1+(l-1))**2+l+m]

    def set_alpha(self, m, l, value):        
        self.alpha[(1+(l-1))**2+l+m] = value

    def set_beta(self, m, l, value):
        self.beta[(1+(l-1))**2+l+m] = value

    def set_gamma(self, m, l, value):
        self.gamma[(1+(l-1))**2+l+m] = value

    def compute_magnetic_field(self):

        self.B_spherical = np.zeros((3,self.n_healpix_pxl))
        self.B_cartesian = np.zeros((3,self.n_healpix_pxl))

        for l in range(self.lmax):
            for m in range(-l, l+1, 1):
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

    def draw_star(self):
        fig, ax = pl.subplots(nrows=1, ncols=5, figsize=(15,6))
        fig.subplots_adjust(left=0.25, bottom=0.25)

        f_vec2pix = lambda x, y, z: hp.pixelfunc.vec2pix(int(self.NSIDE), x, y, z)

        projector = hp.projector.OrthographicProj()

        tmp = projector.projmap(self.B_cartesian[0,:], f_vec2pix, rot=self.los[0,:])[:,0:400]
        l_B1 = ax[0].imshow(tmp)
        tmp = projector.projmap(self.B_cartesian[1,:], f_vec2pix, rot=self.los[0,:])[:,0:400]
        l_B2 = ax[1].imshow(tmp)
        tmp = projector.projmap(self.B_cartesian[2,:], f_vec2pix, rot=self.los[0,:])[:,0:400]
        l_B3 = ax[2].imshow(tmp)
        
        B_los = np.sum(self.B_cartesian * self.los_vec[0,:][:,None], axis=0)
        tmp = projector.projmap(B_los, f_vec2pix)[:,0:400]
        l_B4 = ax[3].imshow(tmp)
        
        
        l_V, = ax[4].plot(self.stokesv[0,:])

        axcolor = 'lightgoldenrodyellow'
        ax_bar = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

        axis = 0
        
        slider = Slider(ax_bar, 'Axis %i index' % axis, 0, self.n_phases-1, valinit=0, valfmt='%i')

        def update(val):
            ind = int(slider.val)
            l_V.set_ydata(self.stokesv[ind,:])

            tmp = projector.projmap(self.B_cartesian[0,:], f_vec2pix, rot=self.los[ind,:])[:,0:400]
            l_B1.set_data(tmp)

            tmp = projector.projmap(self.B_cartesian[1,:], f_vec2pix, rot=self.los[ind,:])[:,0:400]
            l_B2.set_data(tmp)

            tmp = projector.projmap(self.B_cartesian[2,:], f_vec2pix, rot=self.los[ind,:])[:,0:400]
            l_B3.set_data(tmp)

            B_los = np.sum(self.B_cartesian * self.los_vec[0,:][:,None], axis=0)
            tmp = projector.projmap(B_los, f_vec2pix)[:,0:400]            
            l_B4.set_data(tmp)

        slider.on_changed(update)

        pl.show()

    def random_star(self, k, include_toroidal=True):

        # self.set_alpha(1,1,1.0)
        # self.set_alpha(-1,1,1.0)
        # return

        cl = 1.0 / (1.0+(self.l/1.0)**k)
        for l in range(self.lmax+1):                        
            for i in range(2*l+1):
                tmp = np.random.normal(loc=0.0, scale=np.sqrt(cl[l] / 2.0), size=3)
                self.set_alpha(-l+i,l,tmp[0])
                self.set_beta(-l+i,l,tmp[1])
                if (include_toroidal):
                    self.set_gamma(-l+i,l,tmp[2])


    def compute_stellar_spectrum(self, modulus, los, rot_velocity=0.0, nlambda=200, vmax=100, depth=0.5, width=5):
        """
        Compute the averaged spectrum on the star for a given temperature map and for a given rotation
        Args:
            alpha, beta, gamma (float) : arrays of length (1+lmax)**2 with the coefficients for the development of B_radial, B_theta and B_phi
            modulus (float) : modulus of the field
            los (float) : array of size (n_phases,2) with the co-latitude (measured from north pole southward) and longitude of each line-of-sight
            rot_velocity (float) : rotation velocity
        
        Returns:
            stokesi, stokesv (float) : average spectrum and circular polarization
        """
        self.n_phases, _ = los.shape
        self.nlambda = nlambda
        self.v = np.linspace(-vmax, vmax, self.nlambda)
        self.d = depth
        self.delta = width

        self.los = np.zeros_like(los)
        self.los[:,0] = np.rad2deg(los[:,1])
        self.los[:,1] = 90 - np.rad2deg(los[:,0])

        self.los_vec = hp.ang2vec(los[:,0], los[:,1])

        self.stokesi = np.zeros((self.n_phases,self.nlambda))
        self.stokesv = np.zeros((self.n_phases,self.nlambda))

        self.area_projected = np.zeros(self.n_healpix_pxl)

        self.compute_magnetic_field()
        
        for loop in range(self.n_phases):

# Query which pixels are visible
            visible_pixels = hp.query_disc(self.NSIDE, self.los_vec[loop,:], np.pi/2.0-self.epsilon)

# Compute area of each projected pixel
# Option 1: use the fact that all Healpix pixels have the same area of the sphere. If the pixels are small enough, we can
#           just project the area to the plane of the sky computing the scalar product of the normal with the LOS
            self.area_projected = np.sum(self.pixel_vectors * self.los_vec[loop,:][:,None], axis=0)

# Option 2: use a projector to project the boundaries of all pixels on the plane of the sky and 
#           compute the area as a polygon. Slower.
# Define the appropriate projector    
            # projector = hp.projector.OrthographicProj()
            # for pix in visible_pixels:                
            #     tmp = np.array(projector.vec2xy(self.vec_boundaries[:,:,pix]))
            #     self.area_projected[pix] = poly_area(tmp[0,:], tmp[1,:])   

            area = self.area_projected[visible_pixels]
            
            B_los = np.sum(self.B_cartesian * self.los_vec[loop,:][:,None], axis=0)
            vel = np.sum(self.rotation_velocity * self.los_vec[loop,:][:,None], axis=0)

            # vel = self.los_rotation_velocity[visible_pixels]

            total_area = np.sum(area)
                                      
            out = 1.0 - self.d * np.exp(-(self.v[:,None] - rot_velocity * vel[None,visible_pixels])**2 / self.delta**2)
            self.stokesi[loop,:] = np.sum(out * area[None,:], axis=1) / total_area            

            out = 2.0 * self.d * (self.v[:,None] - rot_velocity * vel[None,visible_pixels]) / self.delta**2 * \
                (np.exp(-(self.v[:,None] - rot_velocity * vel[None,visible_pixels])**2 / self.delta**2))

            out *= self.constant * B_los[None,visible_pixels] * modulus

            self.stokesv[loop,:] = np.sum(out * area[None,:], axis=1) / total_area        

        # return np.array([item for sublist in stokesi for item in sublist]), np.array([item for sublist in stokesv for item in sublist])
        return self.stokesi, self.stokesv


if (__name__ == '__main__'):
    NSIDE = 16
    star = synth_gaussian(NSIDE, lmax=2, clv=True)
    n_phases = 20

    los = np.zeros((n_phases,2))
    for i in range(n_phases):
        los[i,:] = np.array([np.pi/2.0, 2.0 * np.pi / n_phases * i])

    star.random_star(k=3.0, include_toroidal=False)

    modulus = 1e3

    stokesi, stokesv = star.compute_stellar_spectrum(modulus, los, rot_velocity=70.0)

    star.draw_star()