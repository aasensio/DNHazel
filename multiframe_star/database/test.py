import numpy as np
import matplotlib.pyplot as pl
from soapy import confParse, SCI, atmosphere
 
# load a sim config that defines lots of science cameras across the field
config = confParse.loadSoapyConfig('sh_8x8.py')

# Init a science camera
sci_camera = SCI.PSF(config, mask=np.ones((154,154)))

# init some atmosphere
atmos = atmosphere.atmos(config)

# Now loop over lots of difference turbulence
for i in range(100):
    print(i)

# Get phase for this time step
    phase_scrns = atmos.moveScrns()

# Calculate all the PSF for this turbulence
    psf = sci_camera.frame(phase_scrns)

# You can also get the phase if you’d like from each science detector with
    # phase = sci_camera.los.phase

pl.imshow(psf)
pl.show()