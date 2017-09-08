import numpy as np
# import matplotlib.pyplot as pl
# import sys
from mpi4py import MPI
from enum import IntEnum
import pyiacsun as ps
import h5py
# from ipdb import set_trace as stop

class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3



nBlocks = 100
nSizeBlock = 1000
n_profiles = nBlocks * nSizeBlock

# Hinode's wavelength axis is 112 in length
hinode_lambda = np.loadtxt('wavelengthHinode.txt')

center = 6301.5080
initial = np.min(hinode_lambda - center) * 1e3
final = np.max(hinode_lambda - center) * 1e3
step = (hinode_lambda[1] - hinode_lambda[0]) * 1e3

lines = [['200,201', initial, step, final]]
n_lambda = ps.radtran.initializeSIR(lines)

hsra = np.loadtxt('hsra_64.model', skiprows=2, dtype='float32')[::-1]

logTau = hsra[:,0]
nz = len(logTau)
model = np.zeros((nz,7), dtype='float32')
model[:,0] = logTau

psf = np.loadtxt('PSF.dat', dtype=np.float32)
ps.radtran.setPSF(psf[:,0].flatten(), psf[:,1].flatten())

def compute(pars):
    n_sizeblock, n_par = pars.shape

    stokesOut = np.zeros((n_sizeblock,4,n_lambda))

    for i in range(n_sizeblock):
        out = ps.radtran.buildModel(logTau, nodes_T=[pars[i,0],pars[i,1],pars[i,2]], nodes_vmic=[pars[i,3]], 
            nodes_B=[pars[i,4],pars[i,5]], nodes_v=[pars[i,6],pars[i,7]], nodes_thB=[pars[i,8],pars[i,9]], 
            nodes_phiB=[pars[i,10],pars[i,11]], var_T=hsra[:,1])

        model[:,1:] = out

        stokes = ps.radtran.synthesizeSIR(model)

        stokesOut[i,:,:] = stokes[1:,:]

    return stokesOut

# T, vmic, B, v, thB, phiB
lower = np.asarray([-3000.0, -1500.0, -3000.0, 0.0, 0.0, 0.0, -7.0, -7.0, 0.0, 0.0, 0.0, 0.0])
upper = np.asarray([3000.0, 3000.0, 5000.0, 4.0, 3000.0, 3000.0, 7.0, 7.0, 180.0, 180.0, 180.0, 180.0])
n_par = 12

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

if rank == 0:

    f = h5py.File('/net/viga/scratch1/deepLearning/DNSIR/database/database_sir.h5', 'w')

    databaseStokes = f.create_dataset("stokes", (n_profiles, 4, n_lambda), dtype='float32')
    databaseLambda = f.create_dataset("lambda", (n_lambda,), dtype='float32')
    databasePars = f.create_dataset("parameters", (n_profiles, n_par))

    databaseLambda[:] = hinode_lambda

    # Master process executes code below
    tasks = []
    for i in range(nBlocks):
        rnd = np.random.rand(nSizeBlock,n_par)
        pars = (upper - lower)[None,:] * rnd + lower[None,:]

        tasks.append(pars)

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
            if task_index < len(tasks):
                dataToSend = {'index': task_index, 'parameters': tasks[task_index]}
                comm.send(dataToSend, dest=source, tag=tags.START)
                print(" * MASTER : sending task {0}/{1} to worker {2}".format(task_index, nBlocks, source), flush=True)
                task_index += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            stokes = dataReceived['stokes']
            index = dataReceived['index']
            pars = dataReceived['parameters']
            
            databaseStokes[index*nSizeBlock:(index+1)*nSizeBlock,:,:] = stokes
            databasePars[index*nSizeBlock:(index+1)*nSizeBlock,:] = pars
    
            print(" * MASTER : got block {0} from worker {1} - saved from {2} to {3}".format(index, source, index*nSizeBlock, (index+1)*nSizeBlock), flush=True)
            
        elif tag == tags.EXIT:
            print(" * MASTER : worker {0} exited.".format(source))
            closed_workers += 1

    print("Master finishing")
    f.close()
else:
    # Worker processes execute code below
    name = MPI.Get_processor_name()    
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
        
        if tag == tags.START:            
            # Do the work here
            task_index = dataReceived['index']
            task = dataReceived['parameters']
            stokes = compute(task)            
            dataToSend = {'index': task_index, 'stokes': stokes, 'parameters': task}
            comm.send(dataToSend, dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)
