import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import h5py
import shutil
from tqdm import tqdm
from ipdb import set_trace as stop

import sys
sys.path.append('../models')

import model1 as model

class dataset_h5(data.Dataset):
    def __init__(self, input_file):
        super(dataset_h5, self).__init__()
        self.input_file = input_file
        self.f = h5py.File(self.input_file, 'r')
        self.images = self.f.get("intensity")
        self.n_training, self.nx, self.ny, self.n_times = self.images.shape
        print(self.images.shape)

    def __getitem__(self, index):
        input = np.transpose(self.images[index,0:48,0:48,:],[2,0,1]).astype('float32')
        target = np.transpose(self.images[index,0:48,0:48,1:2],[2,0,1]).astype('float32')

        return input, target

    def __len__(self):
        return self.n_training


class optical_flow(object):
    def __init__(self, batch_size):
        self.cuda = torch.cuda.is_available()
        self.batch_size = batch_size
        self.model = model.network_syn()
        if (self.cuda):
            self.model.cuda()
            
        self.train_file = '/scratch1/aasensio/deep_learning/optical_flow/database/database_images.h5'
        self.test_file = '/scratch1/aasensio/deep_learning/optical_flow/database/database_images_validation.h5'
            
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
            
        self.test_loader = torch.utils.data.DataLoader(dataset_h5(self.test_file),
            batch_size=self.batch_size, shuffle=True, **kwargs)

        self.checkpoint = '../training/checkpoint.pth.tar'

        print("=> loading checkpoint '{}'".format(self.checkpoint))
        checkpoint = torch.load(self.checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])        
        print("=> loaded checkpoint '{}'".format(self.checkpoint))        


    def test(self):
        self.model.eval()

        data, target = iter(self.test_loader).next()
        
        if self.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, flow = self.model(data)
        out = output.cpu().data.numpy()
        flow = flow.cpu().data.numpy()

        d = data.cpu().data.numpy()
        t = target.cpu().data.numpy()

        f, ax = pl.subplots(nrows=4, ncols=6, figsize=(14,10))
        for i in range(4):
            ax[i,0].imshow(t[i,0,:,:])
            ax[i,1].imshow(out[i,0,:,:])
            ax[i,2].imshow(d[i,0,:,:])
            ax[i,3].imshow(d[i,1,:,:])
            ax[i,4].imshow(flow[i,0,:,:])
            ax[i,5].imshow(flow[i,1,:,:])

        pl.show()
        stop()
        

optical_flow_network = optical_flow(32)
optical_flow_network.test()