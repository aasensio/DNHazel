from __future__ import print_function
import numpy as np
import astropy.convolution

__all__ = ['shiftnd', 'cross_correlation_shifts']

try:
    import fftw3
    has_fftw = True

    def fftwn(array, nthreads=1):
        array = array.astype('complex').copy()
        outarray = array.copy()
        fft_forward = fftw3.Plan(array, outarray, direction='forward',
                                 flags=['estimate'], nthreads=nthreads)
        fft_forward.execute()
        return outarray

    def ifftwn(array, nthreads=1):
        array = array.astype('complex').copy()
        outarray = array.copy()
        fft_backward = fftw3.Plan(array, outarray, direction='backward',
                                  flags=['estimate'], nthreads=nthreads)
        fft_backward.execute()
        return outarray / np.size(array)
except ImportError:
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
    has_fftw = False
# I performed some fft speed tests and found that scipy is slower than numpy
# http://code.google.com/p/agpy/source/browse/trunk/tests/test_ffts.py However,
# the speed varied on machines - YMMV.  If someone finds that scipy's fft is
# faster, we should add that as an option here... not sure how exactly


def get_ffts(nthreads=1, use_numpy_fft=not has_fftw):
    """
    Returns fftn,ifftn using either numpy's fft or fftw
    """
    if has_fftw and not use_numpy_fft:
        def fftn(*args, **kwargs):
            return fftwn(*args, nthreads=nthreads, **kwargs)

        def ifftn(*args, **kwargs):
            return ifftwn(*args, nthreads=nthreads, **kwargs)
    elif use_numpy_fft:
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn
    else:
        # yes, this is redundant, but I feel like there could be a third option...
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn

    return fftn,ifftn

def shiftnd(data, offset, phase=0, nthreads=1, use_numpy_fft=False,
        return_abs=False, return_real=True):
    """
    FFT-based sub-pixel image shift.
    Will turn NaNs into zeros
    Shift Theorem:
    .. math::
        FT[f(t-t_0)](x) = e^{-2 \pi i x t_0} F(x)
    Parameters
    ----------
    data : np.ndarray
        Data to shift
    offset : (int,)*ndim
        Offsets in each direction.  Must be iterable.
    phase : float
        Phase, in radians
    Other Parameters
    ----------------
    use_numpy_fft : bool
        Force use numpy's fft over fftw?  (only matters if you have fftw
        installed)
    nthreads : bool
        Number of threads to use for fft (only matters if you have fftw
        installed)
    return_real : bool
        Return the real component of the shifted array
    return_abs : bool
        Return the absolute value of the shifted array
    Returns
    -------
    The input array shifted by offsets
    """

    fftn,ifftn = get_ffts(nthreads=nthreads, use_numpy_fft=use_numpy_fft)

    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)

    freq_grid = np.sum(
        [off*np.fft.fftfreq(nx)[ 
            [np.newaxis]*dim + [slice(None)] + [np.newaxis]*(data.ndim-dim-1)]
            for dim,(off,nx) in enumerate(zip(offset,data.shape))],
        axis=0)

    kernel = np.exp(-1j*2*np.pi*freq_grid-1j*phase)

    result = ifftn( fftn(data) * kernel )

    if return_real:
        return np.real(result)
    elif return_abs:
        return np.abs(result)
    else:
        return result

def correlate2d(im1,im2, boundary='wrap', **kwargs):
    """
    Cross-correlation of two images of arbitrary size.  Returns an image
    cropped to the largest of each dimension of the input images
    Parameters
    ----------
    return_fft - if true, return fft(im1)*fft(im2[::-1,::-1]), which is the power
        spectral density
    fftshift - if true, return the shifted psd so that the DC component is in
        the center of the image
    pad - Default on.  Zero-pad image to the nearest 2^n
    crop - Default on.  Return an image of the size of the largest input image.
        If the images are asymmetric in opposite directions, will return the largest 
        image in both directions.
    boundary: str, optional
        A flag indicating how to handle boundaries:
            * 'fill' : set values outside the array boundary to fill_value
                       (default)
            * 'wrap' : periodic boundary
    WARNING: Normalization may be arbitrary if you use the PSD
    """

    return astropy.convolution.convolve_fft(np.conjugate(im1), im2[::-1, ::-1], normalize_kernel=False,
            boundary=boundary, ignore_edge_zeros=False, **kwargs)

def cross_correlation_shifts(image1, image2, errim1=None, errim2=None,
                             maxoff=None, verbose=False, gaussfit=False,
                             return_error=False, zeromean=True, **kwargs):
    """ Use cross-correlation and a 2nd order taylor expansion to measure the
    offset between two images
    Given two images, calculate the amount image2 is offset from image1 to
    sub-pixel accuracy using 2nd order taylor expansion.
    Parameters
    ----------
    image1: np.ndarray
        The reference image
    image2: np.ndarray
        The offset image.  Must have the same shape as image1
    errim1: np.ndarray [optional]
        The pixel-by-pixel error on the reference image
    errim2: np.ndarray [optional]
        The pixel-by-pixel error on the offset image.  
    maxoff: int
        Maximum allowed offset (in pixels).  Useful for low s/n images that you
        know are reasonably well-aligned, but might find incorrect offsets due to 
        edge noise
    zeromean : bool
        Subtract the mean from each image before performing cross-correlation?
    verbose: bool
        Print out extra messages?
    gaussfit : bool
        Use a Gaussian fitter to fit the peak of the cross-correlation?
    return_error: bool
        Return an estimate of the error on the shifts.  WARNING: I still don't
        understand how to make these agree with simulations.
        The analytic estimate comes from
        http://adsabs.harvard.edu/abs/2003MNRAS.342.1291Z
        At high signal-to-noise, the analytic version overestimates the error
        by a factor of about 1.8, while the gaussian version overestimates
        error by about 1.15.  At low s/n, they both UNDERestimate the error.
        The transition zone occurs at a *total* S/N ~ 1000 (i.e., the total
        signal in the map divided by the standard deviation of the map - 
        it depends on how many pixels have signal)
    **kwargs are passed to correlate2d, which in turn passes them to convolve.
    The available options include image padding for speed and ignoring NaNs.
    References
    ----------
    From http://solarmuri.ssl.berkeley.edu/~welsch/public/software/cross_cor_taylor.pro
    Examples
    --------
    >>> import numpy as np
    >>> im1 = np.zeros([10,10])
    >>> im2 = np.zeros([10,10])
    >>> im1[4,3] = 1
    >>> im2[5,5] = 1
    >>> import image_registration
    >>> yoff,xoff = image_registration.cross_correlation_shifts(im1,im2)
    >>> im1_aligned_to_im2 = np.roll(np.roll(im1,int(yoff),1),int(xoff),0)
    >>> assert (im1_aligned_to_im2-im2).sum() == 0
    
    """

    if not image1.shape == image2.shape:
        raise ValueError("Images must have same shape.")

    if zeromean:
        image1 = image1 - (image1[image1==image1].mean())
        image2 = image2 - (image2[image2==image2].mean())

    image1 = np.nan_to_num(image1)
    image2 = np.nan_to_num(image2)

    quiet = kwargs.pop('quiet') if 'quiet' in kwargs else not verbose
    ccorr = (correlate2d(image1,image2,quiet=quiet,**kwargs) / image1.size)
    # allow for NaNs set by convolve (i.e., ignored pixels)
    ccorr[ccorr!=ccorr] = 0
    if ccorr.shape != image1.shape:
        raise ValueError("Cross-correlation image must have same shape as input images.  This can only be violated if you pass a strange kwarg to correlate2d.")

    ylen,xlen = image1.shape
    xcen = xlen/2-(1-xlen%2) 
    ycen = ylen/2-(1-ylen%2) 

    if ccorr.max() == 0:
        print("WARNING: No signal found!  Offset is defaulting to 0,0")
        return 0,0

    if maxoff is not None:
        if verbose: print("Limiting maximum offset to %i" % maxoff)
        subccorr = ccorr[ycen-maxoff:ycen+maxoff+1,xcen-maxoff:xcen+maxoff+1]
        ymax,xmax = np.unravel_index(subccorr.argmax(), subccorr.shape)
        xmax = xmax+xcen-maxoff
        ymax = ymax+ycen-maxoff
    else:
        ymax,xmax = np.unravel_index(ccorr.argmax(), ccorr.shape)
        subccorr = ccorr

    if return_error:
        if errim1 is None:
            errim1 = np.ones(ccorr.shape) * image1[image1==image1].std() 
        if errim2 is None:
            errim2 = np.ones(ccorr.shape) * image2[image2==image2].std() 
        eccorr =( (correlate2d(errim1**2, image2**2,quiet=quiet,**kwargs)+
                   correlate2d(errim2**2, image1**2,quiet=quiet,**kwargs))**0.5 
                   / image1.size)
        if maxoff is not None:
            subeccorr = eccorr[ycen-maxoff:ycen+maxoff+1,xcen-maxoff:xcen+maxoff+1]
        else:
            subeccorr = eccorr

    if gaussfit:
        try: 
            from agpy import gaussfitter
        except ImportError:
            raise ImportError("Couldn't import agpy.gaussfitter; try using cross_correlation_shifts with gaussfit=False")
        if return_error:
            pars,epars = gaussfitter.gaussfit(subccorr,err=subeccorr,return_all=True)
            exshift = epars[2]
            eyshift = epars[3]
        else:
            pars,epars = gaussfitter.gaussfit(subccorr,return_all=True)
        xshift = maxoff - pars[2] if maxoff is not None else xcen - pars[2]
        yshift = maxoff - pars[3] if maxoff is not None else ycen - pars[3]
        if verbose: 
            print("Gaussian fit pars: ",xshift,yshift,epars[2],epars[3],pars[4],pars[5],epars[4],epars[5])

    else:

        xshift_int = xmax-xcen
        yshift_int = ymax-ycen

        local_values = ccorr[ymax-1:ymax+2,xmax-1:xmax+2]

        d1y,d1x = np.gradient(local_values)
        d2y,d2x,dxy = second_derivative(local_values)

        fx,fy,fxx,fyy,fxy = d1x[1,1],d1y[1,1],d2x[1,1],d2y[1,1],dxy[1,1]

        shiftsubx=(fyy*fx-fy*fxy)/(fxy**2-fxx*fyy)
        shiftsuby=(fxx*fy-fx*fxy)/(fxy**2-fxx*fyy)

        xshift = -(xshift_int+shiftsubx)
        yshift = -(yshift_int+shiftsuby)

        # http://adsabs.harvard.edu/abs/2003MNRAS.342.1291Z
        # Zucker error

        if return_error:
            #acorr1 = (correlate2d(image1,image1,quiet=quiet,**kwargs) / image1.size)
            #acorr2 = (correlate2d(image2,image2,quiet=quiet,**kwargs) / image2.size)
            #ccorrn = ccorr / eccorr**2 / ccorr.size #/ (errim1.mean()*errim2.mean()) #/ eccorr**2
            normalization = 1. / ((image1**2).sum()/image1.size) / ((image2**2).sum()/image2.size) 
            ccorrn = ccorr * normalization
            exshift = (np.abs(-1 * ccorrn.size * fxx*normalization/ccorrn[ymax,xmax] *
                    (ccorrn[ymax,xmax]**2/(1-ccorrn[ymax,xmax]**2)))**-0.5) 
            eyshift = (np.abs(-1 * ccorrn.size * fyy*normalization/ccorrn[ymax,xmax] *
                    (ccorrn[ymax,xmax]**2/(1-ccorrn[ymax,xmax]**2)))**-0.5) 
            if np.isnan(exshift):
                raise ValueError("Error: NAN error!")

    if return_error:
        return xshift,yshift,exshift,eyshift
    else:
        return yshift,xshift

def second_derivative(image):
    """
    Compute the second derivative of an image
    The derivatives are set to zero at the edges
    Parameters
    ----------
    image: np.ndarray
    Returns
    -------
    d/dx^2, d/dy^2, d/dxdy
    All three are np.ndarrays with the same shape as image.
    """
    shift_right = np.roll(image,1,1)
    shift_right[:,0] = 0
    shift_left = np.roll(image,-1,1)
    shift_left[:,-1] = 0
    shift_down = np.roll(image,1,0)
    shift_down[0,:] = 0
    shift_up = np.roll(image,-1,0)
    shift_up[-1,:] = 0

    shift_up_right = np.roll(shift_up,1,1)
    shift_up_right[:,0] = 0
    shift_down_left = np.roll(shift_down,-1,1)
    shift_down_left[:,-1] = 0
    shift_down_right = np.roll(shift_right,1,0)
    shift_down_right[0,:] = 0
    shift_up_left = np.roll(shift_left,-1,0)
    shift_up_left[-1,:] = 0

    dxx = shift_right+shift_left-2*image
    dyy = shift_up   +shift_down-2*image
    dxy=0.25*(shift_up_right+shift_down_left-shift_up_left-shift_down_right)

    return dxx,dyy,dxy


if (__name__ == '__main__'):
    import matplotlib.pyplot as pl
    x = np.linspace(-4.0,4.0,100)
    y = np.linspace(-4.0,4.0,100)
    X, Y = np.meshgrid(x,y)
    im = np.exp(-X**2-Y**2)
    im2 = np.roll(np.roll(im, 30, axis=0), 23, axis=1)

    shf = cross_correlation_shifts(im, im2)

    res2 = shiftnd(im, shf)

    f, ax = pl.subplots(nrows=2, ncols=1)
    ax[0].imshow(im2)
    ax[1].imshow(res2)

    pl.show()