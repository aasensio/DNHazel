import numpy as np
import matplotlib.pyplot as pl
import poppy
import psf
from astropy import units as u


telescope_radius = 0.5 * u.meter
pixel_size = 0.03 * u.arcsec / u.pixel
fov = (101, 101) * u.pixel            
lambda0 = 500 * u.nm

# In-focus PSF
osys = poppy.OpticalSystem()

osys.add_pupil(poppy.CircularAperture(radius = telescope_radius))
osys.add_detector(pixelscale=pixel_size, fov_pixels=fov, oversample=1)

psf_poppy = osys.calc_psf(wavelength=lambda0)

osys2 = psf.psf(2*telescope_radius.value, pixel_size.value, lambda0.to(u.AA).value, 101, 10)
osys2.compute_aperture(centralObs = 0, spider = 0)
psf_mine = osys2.center_image(osys2.get_psf_diffraction())

x = np.linspace(-101*0.03*0.5,101*0.03*0.5, 101)
diff = 206265.0 * 1.22 * (lambda0.to(u.m) / (2*telescope_radius))

pl.close('all')
pl.semilogy(x, psf_poppy[0].data[:,50] / np.max(psf_poppy[0].data))
pl.semilogy(x, psf_mine[:,50] / np.max(psf_mine))
pl.axvline(diff)
pl.show()

osys2.init_covariance(0.1)
res = osys2.center_image(osys2.get_psf_seeing(no_piston=True, no_tiptilt=True))