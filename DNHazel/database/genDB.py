import numpy as np
import sys
from mpi4py import MPI
from enum import IntEnum
import pyhazel
import pyiacsun as ps
from scipy.io import netcdf

class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

def compute(pars):
    nPar, nSizeBlock = pars.shape

    stokesOut = np.zeros((4,128,nSizeBlock))

    nLambdaInput = 128
    GRIS_dispersion = 0.0362  # A/pix
    lowerLambda = 10828
    upperLambda = lowerLambda + GRIS_dispersion * nLambdaInput

    for i in range(nSizeBlock):
        tau, v, vth, a, B, theta, phi, mu = pars[:,i]

        synModeInput = 5
        nSlabsInput = 1

        B1Input = np.asarray([B, theta, phi])    
        B2Input = np.asarray([0.0,0.0,0.0])
        
        hInput = 3.e0

        tau1Input = tau
        tau2Input = 0.e0

        I0 = ps.hazel.i0Allen(10830.0, mu)

        boundaryInput  = np.asarray([I0,0.0,0.0,0.0])

        transInput = 1
        atomicPolInput = 1

        anglesInput = np.asarray([0.0,0.0,0.0])

        lambdaAxisInput = np.linspace(lowerLambda-10829.0911, upperLambda-10829.0911, nLambdaInput)        

        dopplerWidthInput = vth
        dopplerWidth2Input = 0.e0

        dampingInput = a

        dopplerVelocityInput = v
        dopplerVelocity2Input = 0.e0

        ffInput = 0.e0
        betaInput = 1.0
        beta2Input = 1.0
        nbarInput = np.asarray([0.0,0.0,0.0,0.0])
        omegaInput = np.asarray([0.0,0.0,0.0,0.0])
        
        nbarInput = np.asarray([1.0,1.0,1.0,1.0])
        omegaInput = np.asarray([1.0,1.0,1.0,1.0])
        normalization = 0
        
        # Compute the Stokes parameters using many default parameters, using Allen's data
        [l, stokes, etaOutput, epsOutput] = pyhazel.synth(synModeInput, nSlabsInput, B1Input, B2Input, hInput, 
                                tau1Input, tau2Input, boundaryInput, transInput, atomicPolInput, anglesInput, 
                                nLambdaInput, lambdaAxisInput, dopplerWidthInput, dopplerWidth2Input, dampingInput, 
                                dopplerVelocityInput, dopplerVelocity2Input, ffInput, betaInput, beta2Input, nbarInput, omegaInput, normalization)

        stokesOut[:,:,i] = stokes

    return l, stokesOut

nBlocks = 1000
nSizeBlock = 100
nProfiles = nBlocks * nSizeBlock

outputFile = 'database_mu1.db'

# tau, v, vth, a, B, theta, phi
lower = np.asarray([0.05, -5.0, 5.0, 0.0, 0.0, 0.0, -180.0, 1.0])
upper = np.asarray([3.0, 5.0, 18.0, 0.5, 1000.0, 180.0, 180.0, 1.0])
nPar = 8

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

pyhazel.init()

if rank == 0:

    f = netcdf.netcdf_file(outputFile, 'w')
    f.history = 'Database of profiles'
    f.createDimension('nProfiles', nProfiles)
    f.createDimension('nLambda', 128)
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