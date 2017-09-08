import numpy as np
import matplotlib.pyplot as pl
import scipy.io as io
from PIL import Image, ImageDraw
import glob
import h5py
from ipdb import set_trace as stop
from tqdm import tqdm
from skimage import exposure
from scipy.interpolate import interp1d
from photutils import Background2D, SigmaClip, MedianBackground

def random_position_circle(mask, side_len):
    """
    Get random patches on the Sun making sure that at least one part of a filament is inside
    """
    coef = 2048 - side_len
    rnd = (coef * np.random.rand(2)).astype('int')
    square = mask[rnd[0]:rnd[0]+side_len,rnd[1]:rnd[1]+side_len]
    
    while ((mask[rnd[0],rnd[1]] == 0) or (np.max(square) != 2)):
        rnd = (coef * np.random.rand(2)).astype('int')
        square = mask[rnd[0]:rnd[0]+side_len,rnd[1]:rnd[1]+side_len]

    return rnd[0], rnd[1]

def perturb_patch(image, mask):
    """
    Augment the patch by random rotations and flips
    """
    angle = np.random.randint(0, 3)
    flipx = np.random.randint(0, 1)
    flipy = np.random.randint(0, 1)
    
    image_new = np.rot90(image, angle)
    mask_new = np.rot90(mask, angle)

    if (flipx == 1):
        image_new = np.flip(image_new, 0)
        mask_new = np.flip(mask_new, 0)

    if (flipy == 1):
        image_new = np.flip(image_new, 1)
        mask_new = np.flip(mask_new, 1)

    return image_new, mask_new

def centered_distance_matrix(n):
    # make sure n is odd
    x,y = np.meshgrid(range(n),range(n))
    return np.sqrt((x-(n/2)+1)**2+(y-(n/2)+1)**2)

def function(d):
    return np.log(d) # or any funciton you might have

def arbitraryfunction(d,y):
    x = np.arange(len(y))
    f = interp1d(x, y, fill_value='extrapolate')
    return f(d.flat).reshape(d.shape)

def radial_profile(data, center):
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    d = centered_distance_matrix(2048)
    f = arbitraryfunction(d,radialprofile)

    return f


nx = 128
ny = 128

files = glob.glob('*.save')
n_files = len(files)

im_ha = []
mask_ha = []
mask_disk = []


for ff in files:
    print("Correcting image {0}".format(ff))
    res = io.readsav(ff)

    _, n_filaments = res['xcoord'].shape

    dx = res['map']['dx'][0]
    dy = res['map']['dy'][0]
    xc = res['map']['xc'][0]
    yc = res['map']['yc'][0]
    rsun = res['map']['rsun'][0]
    im = res['map'][0][0]

    print(" - Background correction")
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(im, (30, 30), filter_size=(3, 3),
        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    im = im / bkg.background

    # stop()

    # radial = radial_profile(im,center=[1024,1024])
    # im = np.nan_to_num(im / radial)

    print(" - Mask calculation")
    mask = np.zeros_like(im)
    x_map = (np.arange(2048) - 1024) * dx + xc
    y_map = (np.arange(2048) - 1024) * dy + yc
    X, Y = np.meshgrid(x_map, y_map)
    mask[np.sqrt(X**2+Y**2) < rsun] = 1.0

    mask_disk.append(mask)

    # im = exposure.equalize_hist(im)

    im *= mask    

    im_filaments = Image.new('L', (2048, 2048), color=1)

    print(" - Detecting filaments")

    for i in range(n_filaments):
        x = res['xcoord'][:,i]
        y = res['ycoord'][:,i]

        x = x[x != -1e7][1:]
        y = y[y != -1e7][1:]

        x = np.append(x, x[0])
        y = np.append(y, y[0])

        x = (x-xc) / dx + 1024
        y = (y-yc) / dy + 1024

        polygon = np.dstack((x, y)).flatten()

        ImageDraw.Draw(im_filaments).polygon(polygon, outline=1, fill=2)

    mask_filaments = np.array(im_filaments)
    mask_filaments = mask * mask_filaments

    im_ha.append(im)
    mask_ha.append(mask_filaments)

mn, mx = np.min(np.array(im_ha)[:,mask==1]), np.max(np.array(im_ha)[:,mask==1])

np.save('images_ha.npy', im_ha)
np.save('mask_ha.npy', mask_ha)
np.save('normalization.npy', [mn, mx])

n_images_per_map = 1000
n_patches = n_files * n_images_per_map

f_training = h5py.File('/net/viga/scratch1/deepLearning/mluna_segmentation/database/database.h5', 'w')
db_images = f_training.create_dataset('halpha', (n_patches, nx, ny, 1), 'f')
db_mask = f_training.create_dataset('mask', (n_patches, nx, ny, 1), 'f')

loop = 0
ind = np.random.permutation(np.arange(n_patches))

for j in range(n_files):
    for i in tqdm(range(n_images_per_map)):
        x_pos, y_pos = random_position_circle(mask_ha[j], nx)

        im_patch = (im_ha[j][x_pos:x_pos+nx, y_pos:y_pos+ny] - mn) / (mx - mn)
        mask_patch = mask_ha[j][x_pos:x_pos+nx, y_pos:y_pos+ny]

        im_augmented, mask_augmented = perturb_patch(im_patch, mask_patch)

        db_images[ind[loop],:,:,0] = im_augmented
        db_mask[ind[loop],:,:,0] = mask_augmented
        loop += 1

f_training.close()

n_images_per_map = 100
n_patches = n_files * n_images_per_map

f_validation = h5py.File('/net/viga/scratch1/deepLearning/mluna_segmentation/database/database_validation.h5', 'w')
db_images = f_validation.create_dataset('halpha', (n_patches, nx, ny, 1), 'f')
db_mask = f_validation.create_dataset('mask', (n_patches, nx, ny, 1), 'f')

loop = 0
ind = np.random.permutation(np.arange(n_patches))
for j in range(n_files):
    for i in tqdm(range(n_images_per_map)):
        x_pos, y_pos = random_position_circle(mask_ha[j], nx)

        im_patch = (im_ha[j][x_pos:x_pos+nx, y_pos:y_pos+ny] - mn) / (mx - mn)
        mask_patch = mask_ha[j][x_pos:x_pos+nx, y_pos:y_pos+ny]

        im_augmented, mask_augmented = perturb_patch(im_patch, mask_patch)

        db_images[ind[loop],:,:,0] = im_augmented
        db_mask[ind[loop],:,:,0] = mask_augmented
        loop += 1

f_validation.close()