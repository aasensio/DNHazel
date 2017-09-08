import numpy as np
import healpy as hp
import synth_gaussian as synth
from mpi4py import MPI
from enum import IntEnum
import h5py

class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

n_batches = 300
batchsize = 16
n_stars = n_batches * batchsize
nlambda = 150

def compute(star, nlambda):

    alpha = np.zeros((batchsize, star.n_coef_max))
    beta = np.zeros((batchsize, star.n_coef_max))
    gamma = np.zeros((batchsize, star.n_coef_max))

    n_angles = np.random.randint(low=10, high=30)

    los = np.zeros((n_angles,2))
    for i in range(n_angles):
        los[i,:] = np.array([np.pi/2.0, 2.0 * np.pi / n_angles * i])

    stokesv = np.zeros((batchsize, n_angles * nlambda))
    angles = np.zeros((batchsize, n_angles * 2))

    modulus = np.random.uniform(low=200.0, high=2500.0, size=batchsize)

    for i in range(batchsize):
        star.random_star(k=3.0, include_toroidal=True)
        
        si, sv = star.compute_stellar_spectrum(modulus[i], los, rot_velocity=70.0, nlambda=nlambda)

        alpha[i,:] = star.alpha
        beta[i,:] = star.beta
        gamma[i,:] = star.gamma
        angles[i,:] = los.flatten()
        stokesv[i,:] = sv.flatten()

    return alpha, beta, gamma, stokesv, angles, modulus

def master_work(filename, n_batches, batchsize, nlambda):
    f = h5py.File(filename, 'w')

    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    database_stokesv = f.create_dataset("stokesv", (n_stars,), dtype=dt)
    database_alpha = f.create_dataset("alpha", (n_stars,), dtype=dt)
    database_beta = f.create_dataset("beta", (n_stars,), dtype=dt)
    database_gamma = f.create_dataset("gamma", (n_stars,), dtype=dt)
    database_angles = f.create_dataset("angles", (n_stars,), dtype=dt)
    database_modulus = f.create_dataset("modulus", (n_stars,), dtype=dt)
        
    task_index = 0
    num_workers = size - 1
    closed_workers = 0
    print("*** Master starting with {0} workers".format(num_workers))
    while closed_workers < num_workers:
        dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)                
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
                # Worker is ready, so send it a task
            if task_index < n_batches:
                dataToSend = {'index': task_index} #, 'parameters': tasks[task_index]}
                comm.send(dataToSend, dest=source, tag=tags.START)
                print(" * MASTER : sending task {0}/{1} to worker {2}".format(task_index, n_batches, source), flush=True)
                task_index += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            index = dataReceived['index']
            alpha = dataReceived['alpha']
            beta = dataReceived['beta']
            gamma = dataReceived['gamma']                
            stokesv = dataReceived['stokesv']
            angles = dataReceived['angles']
            modulus = dataReceived['modulus']
                                
            for i in range(batchsize):
                database_stokesv[index*batchsize + i] = stokesv[i,:]
                database_alpha[index*batchsize + i] = alpha[i,:]
                database_beta[index*batchsize + i] = beta[i,:]
                database_gamma[index*batchsize + i] = gamma[i,:]                    
                database_angles[index*batchsize + i] = angles[i,:]
                database_modulus[index*batchsize + i] = np.atleast_1d(modulus[i])
                
            print(" * MASTER : got block {0} from worker {1} - saved {2} - nangles={3}".format(index, source, index, len(angles[0,:])), flush=True)
                
        elif tag == tags.EXIT:
            print(" * MASTER : worker {0} exited.".format(source))
            closed_workers += 1

    print("Master finishing")
    f.close()

def slave_work():

    star = synth.synth_gaussian(16, lmax=3, clv=True)

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
            
        if tag == tags.START:            
                # Do the work here
            task_index = dataReceived['index']

            alpha, beta, gamma, stokesv, angles, modulus = compute(star, nlambda)
                
            dataToSend = {'index': task_index, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'stokesv': stokesv, 'angles': angles, 'modulus': modulus}
            comm.send(dataToSend, dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)


if (__name__ == '__main__'):
    
# Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    if rank == 0:

        n_batches = 300
        batchsize = 16
        n_stars = n_batches * batchsize
        nlambda = 150
        filename = '/net/viga/scratch1/deepLearning/doppler_imaging/database/validation_stars.h5'

        master_work(filename, n_batches, batchsize, nlambda)

        # n_batches = 3000
        # batchsize = 16
        # n_stars = n_batches * batchsize
        # nlambda = 150
        # filename = '/net/viga/scratch1/deepLearning/doppler_imaging/database/training_stars.h5'
        # master_work(filename, n_batches, batchsize, nlambda)
        
    else:
        slave_work()
        # slave_work()
        # Worker processes execute code below
        
