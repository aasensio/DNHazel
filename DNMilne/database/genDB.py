import numpy as np
import sys
from mpi4py import MPI
from enum import IntEnum
import pyiacsun as ps
from scipy.io import netcdf

class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

def compute(pars):
    nPar, nSizeBlock = pars.shape

    stokesOut = np.zeros((4,64,nSizeBlock))

    stokesOut = milne.synthGroup(pars)

    return milne.wavelength, stokesOut

nBlocks = 1000
nSizeBlock = 1000
nProfiles = nBlocks * nSizeBlock

lambda0 = 6301.5080
JUp = 2.0
JLow = 2.0
gUp = 1.5
gLow = 1.833
lambdaStart = 6300.8
lambdaStep = 0.03
nLambda = 64

lineInfo = np.asarray([lambda0, JUp, JLow, gUp, gLow, lambdaStart, lambdaStep])
milne = ps.radtran.milne(nLambda, lineInfo)

# BField, theta, chi, vmac, damping, B0, B1, doppler, kl
lower = np.asarray([0.0,      0.0,   0.0, -6.0, 0.0,  0.1, 0.1, 0.045,  0.1])
upper = np.asarray([3000.0, 180.0, 180.0,  6.0, 0.5, 20.0, 20.0, 0.100, 20.0])
nPar = 9

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

if rank == 0:

    f = netcdf.netcdf_file('/scratch1/deepLearning/DNMilne/database/database_1000000.db', 'w')
    f.history = 'Database of profiles'
    f.createDimension('nProfiles', nProfiles)
    f.createDimension('nLambda', nLambda)
    f.createDimension('nStokes', 4)
    f.createDimension('nParameters', nPar)
    databaseStokes = f.createVariable('stokes', 'f', ('nStokes', 'nLambda', 'nProfiles'))
    databaseLambda = f.createVariable('lambda', 'f', ('nLambda',))
    databasePars = f.createVariable('parameters', 'f', ('nParameters','nProfiles'))

    # Master process executes code below
    tasks = []
    for i in range(nBlocks):
        rnd = np.random.rand(nPar, nSizeBlock)
        pars = (upper - lower)[:,None] * rnd + lower[:,None]

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
            l = dataReceived['lambda']
            pars = dataReceived['parameters']
            
            databaseStokes[:,:,index*nSizeBlock:(index+1)*nSizeBlock] = stokes
            databasePars[:,index*nSizeBlock:(index+1)*nSizeBlock] = pars

            if (index % 100 == 0):
                f.flush()

            if (index == 0):
                databaseLambda[:] = l
            
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
            l, stokes = compute(task)            
            dataToSend = {'index': task_index, 'stokes': stokes, 'lambda': l, 'parameters': task}
            comm.send(dataToSend, dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)
