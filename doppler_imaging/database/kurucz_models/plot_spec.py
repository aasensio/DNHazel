import numpy as np
import matplotlib.pyplot as pl
import struct
import astropy.constants as const
import scipy.interpolate

def read_kurucz_spec(f):
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

    wavelength *= 1e8

    w_air = wavelength / (1.0 + 2.735182e-4 + 131.4182 / wavelength**2 + 2.76249e8 / wavelength**4)

# # Reinterpolate in a equidistant velocity axis
#     new_vel = np.linspace(np.min(vel), np.max(vel), nl)
#     for i in range(nmus):
#         interpolator = scipy.interpolate.interp1d(vel, stokes[:,i], kind='linear')
#         stokes[:,i] = interpolator(new_vel)

    return wavelength, stokes, cont

if (__name__ == '__main__'):
    lam, spec, cont = read_kurucz_spec('RESULTS/T_9000_logg4.0_feh0.0.spec')
    pl.plot(lam, spec[:,0] / cont[:,0])
    pl.show()
