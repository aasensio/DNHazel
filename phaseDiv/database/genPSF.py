import numpy as np
import matplotlib.pyplot as pl
import wavefront as wf

try:
	import progressbar as pb
	doProgress = True
except:
	doProgress = False
	pass

class pyPSF(object):

	def __init__(self, telescopeDiameter, pixSize, lambda0, nPixPSF):
		"""
		Initialization of the class
		
		Args:			 
			 telescopeDiameter (real): telescope diameter in m
			 pixSize (real): pixel size in arcsec
			 lambda0 (real): observing wavelength
			 nPixPSF (int, optional): number of pixels where to sample the PSF
		
		Returns:
			 TYPE: Description
		"""		
		self.lambda0 = lambda0
		self.telescopeDiameter = telescopeDiameter
		self.pixSize = pixSize		
		self.nPixPSF = nPixPSF

	def computeAperture(self, centralObs = 0, spider = 0):
		"""
		Compute the aperture of the telescope
		
		Args:
			 centralObs (int, optional): central obscuration
			 spider (int, optional): spider size in pixel
		
		Returns:
			 real: compute the aperture
		"""
		self.aperture = wf.aperture(npix = self.nPixPSF, cent_obs = centralObs, spider=spider)


	def generateSeeingPSF(self, r0, nterms, defocus=None):
		"""
		Compute a seeing PSF
		
		Args:
			 r0 (real): Fried parameter [cm]
			 nterms (int): number of terms to include in the wavefront generation
		
		Returns:
			 real: the final PSF
		"""
		print(wf.psfScale(self.telescopeDiameter, self.lambda0, self.pixSize))
		self.wavefront = wf.seeing(self.telescopeDiameter * 100.0 / r0, npix = self.nPixPSF, nterms = nterms, quiet=True, defocus=defocus)		
		self.psf = wf.psf(self.aperture, self.wavefront, overfill = wf.psfScale(self.telescopeDiameter, self.lambda0, self.pixSize))

# Pad the PSF image to make it equal to the original image		
		self.psf = np.roll(self.psf, int(self.nPixPSF/2), axis=0)
		self.psf = np.roll(self.psf, int(self.nPixPSF/2), axis=1)
		return self.psf

	def centerImage(self, image):
		size = image.shape
		return np.roll(np.roll(image, int(size[0]/2), axis=0), int(size[1]/2), axis=1)

	def convolvePSF(self, index):
		self.psfFFT = np.fft.fft2(self.psf)
		return np.real(np.fft.ifft2(self.psfFFT * np.fft.fft2(self.cube[:,:,index])))

if (__name__ == '__main__'):
	# GREGOR
	telescopeDiameter = 1.440      # m
	secondaryDiameter = 0.404      # m
	pixSize = 0.126              # arcsec
	lambda0 = 10830.0            # Angstrom
	spatialResol = 0.5           # arcsec
	r0 = 1.22 * lambda0 * 206265. / spatialResol * 1e-8                   # computed from the spatial resolution [in cm]

	pl.close('all')

	nPix = 65
	nRealizations = 50

	# Initialize the class with the telescope information
	out = pyPSF(telescopeDiameter, pixSize, lambda0, nPix)

	# Compute aperture and return the diffraction PSF
	out.computeAperture(centralObs = (secondaryDiameter/telescopeDiameter)**2, spider = 0)
	psfDiffraction = out.centerImage(out.generateSeeingPSF(r0, nterms=0))

	# Now generate many realizations of the seeing and average them out to
	# mimick a long integration time observation. We use 20 random Zernike modes
	psf = np.zeros((nPix,nPix))
	if (doProgress):
		pbar = pb.ProgressBar(maxval=nRealizations).start()
	for i in range(nRealizations):
		if (doProgress):
			pbar.update(i+1)
		psf += out.generateSeeingPSF(r0, nterms=20)	

	if (doProgress):
		pbar.finish()

	psf /= (1.0*nRealizations)
	psfCenter = out.centerImage(psf)

	f, ax = pl.subplots(ncols=2, nrows=1, figsize=(10,8))
	ax[0].imshow(np.log(psfCenter))
	ax[1].imshow(np.log(psfDiffraction))

	# Save the PSF for later use
	# np.save('psfVTTDiffraction_pix0.36.npy', psfDiffraction)
	# np.save('psfVTT0.5arcsec_pix036.npy', psf)

	# np.save('psfGREGORDiffraction_10830.npy', psfDiffraction)
	# np.save('psfGREGORSeeing0.5_10830.npy', psf)