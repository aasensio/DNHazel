import numpy as np
import matplotlib.pyplot as pl
import h5py
import scipy.io as io
import matplotlib.animation as manimation
from ipdb import set_trace as stop

f = h5py.File('imax_velocity.h5')
vel = f.get("velocity")
n_frames, nx_vel, ny_vel, _ = vel.shape

im = io.readsav('cont.idl')['cont'][:,100:100+nx_vel,100:100+ny_vel]
_, nx, ny = im.shape

x = np.linspace(0, nx_vel, nx_vel/8)
y = np.linspace(0, ny_vel, ny_vel/8)
X, Y = np.meshgrid(x, y)

pl.close('all')
# f, ax = pl.subplots(figsize=(15,12))

x = np.arange(800)
y = np.arange(800)
X, Y = np.meshgrid(x, y)

# ax.imshow(im[0,0:100,0:100])
# ax.quiver(X[0:100,0:100], Y[0:100,0:100], vel[0,0:100,0:100,0], vel[0,0:100,0:100,1], angles='xy')
# ax.streamplot(X[0:100,0:100], Y[0:100,0:100], vel[0,0:100,0:100,0], vel[0,0:100,0:100,1])
# pl.show()

# div = np.diff(vel[0,:,:-1,0], axis=0) + np.diff(vel[0,:-1,:,1], axis=1)
# ax.imshow(im[0,0:160,0:160])

# ax.quiver(X[0:20,0:20], Y[0:20,0:20], vel[0,::8,::8,0][0:20,0:20], vel[0,::8,::8,1][0:20,0:20], headwidth=3, headlength=5, scale=30, color='r')

# pl.show()

# stop()

print("Plotting validation data...")

label = ['1', '0.1', '0.01']

n_frames = 55

# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
# writer = FFMpegWriter(fps=3, bitrate=10000, metadata=metadata)
# with writer.saving(f, "movie.mp4", n_frames):
#     for i in range(n_frames):
#         print ("Frame {0}/{1}".format(i,n_frames))

#         div = np.diff(vel[i,:,:-1,0], axis=0) + np.diff(vel[i,:-1,:,1], axis=1)
        
#         ax.imshow(im[i,:,:], cmap=pl.cm.gray)
#         # ax.imshow(div)
#         ax.quiver(X[::8,::8], Y[::8,::8], vel[i,::8,::8,0], vel[i,::8,::8,1], headwidth=3, headlength=5, scale=30, color='r', angles='xy')
#         ax.set_title(r'$\tau=${0}'.format(label[0]))

#         writer.grab_frame()
#         ax.cla()

# f, ax = pl.subplots(ncols=3, nrows=1, figsize=(18,10))

# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
# writer = FFMpegWriter(codec='libx264', fps=3, bitrate=20000, metadata=metadata)
# with writer.saving(f, "movie_small.mp4", n_frames):
#     for i in range(n_frames):
#         print ("Frame {0}/{1}".format(i,n_frames))

#         for j in range(3):
#             ax[j].imshow(im[i,0:200,0:200], cmap=pl.cm.gray)
#             ax[j].quiver(X[::8,::8][0:25,0:25], Y[::8,::8][0:25,0:25], vel[i,::8,::8,2*j][0:25,0:25], vel[i,::8,::8,2*j+1][0:25,0:25], headwidth=3, headlength=5, scale=20, color='r', angles='xy')
#             ax[j].set_title(r'$\tau=${0}'.format(label[j]))

#         writer.grab_frame()
#         for j in range(3):
#             ax[j].cla()


# f, ax = pl.subplots(ncols=3, nrows=1, figsize=(18,10))

# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
# writer = FFMpegWriter(codec='libx264', fps=3, bitrate=20000, metadata=metadata)
# with writer.saving(f, "movie_verysmall.mp4", n_frames):
#     for i in range(n_frames):
#         print ("Frame {0}/{1}".format(i,n_frames))

#         for j in range(3):
#             ax[j].imshow(im[i,0:100,0:100], cmap=pl.cm.gray)
#             ax[j].quiver(X[0:100,0:100], Y[0:100,0:100], vel[i,0:100,0:100,2*j], vel[i,0:100,0:100,2*j+1], headwidth=3, headlength=5, scale=20, color='r', angles='xy')
#             ax[j].set_title(r'$\tau=${0}'.format(label[j]))

#         writer.grab_frame()
#         for j in range(3):
#             ax[j].cla()

f, ax = pl.subplots()
start = np.vstack([X[::8,::8][0:25,0:25].flatten(), Y[::8,::8][0:25,0:25].flatten()])

n_frames=55

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
writer = FFMpegWriter(codec='libx264', fps=3, bitrate=20000, metadata=metadata)
with writer.saving(f, "movie_stream.mp4", n_frames):
    for i in range(n_frames):
        print ("Frame {0}/{1}".format(i,n_frames))

        div = np.diff(vel[i,:,:-1,0], axis=0) + np.diff(vel[i,:-1,:,1], axis=1)
        
        ax.imshow(im[i,0:200,0:200], cmap=pl.cm.gray)        
        ax.quiver(X[::8,::8][0:25,0:25], Y[::8,::8][0:25,0:25], vel[i,::8,::8,0][0:25,0:25], vel[i,::8,::8,1][0:25,0:25], headwidth=3, headlength=5, scale=20, color='r', angles='xy')
        ax.streamplot(x[::8][0:25], y[::8][0:25], vel[i,::8,::8,0][0:25,0:25], vel[i,::8,::8,1][0:25,0:25], start_points=start.T)
        ax.set_title(r'$\tau=${0}'.format(label[0]))
        ax.set_xlim([0,200])
        ax.set_ylim([0,200])

        writer.grab_frame()
        ax.cla()