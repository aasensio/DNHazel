import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import h5py
import shutil
from tqdm import tqdm

import sys
sys.path.append('../models')

import model1 as model

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class dataset_h5(torch.utils.data.Dataset):
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
                    
        self.train_file = '/scratch1/aasensio/deep_learning/optical_flow/database/database_images.h5'
        self.test_file = '/scratch1/aasensio/deep_learning/optical_flow/database/database_images_validation.h5'
            
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        
        self.train_loader = torch.utils.data.DataLoader(dataset_h5(self.train_file),
            batch_size=self.batch_size, shuffle=True, **kwargs)
        
        self.test_loader = torch.utils.data.DataLoader(dataset_h5(self.test_file),
            batch_size=self.batch_size, shuffle=True, **kwargs)
        
        self.model = model.network()
        if self.cuda:
            self.model.cuda()
            
    def optimize(self, epochs, lr=1e-4):
        
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
        if self.cuda:
            self.loss_fn.cuda()
        
        self.loss = []
        self.loss_val = []
        best_loss = 1e10
        
        for epoch in range(1, epochs + 1):
            self.train(epoch)
            self.test()
            
            is_best = self.loss_val[-1] > best_loss
            best_loss = max(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                #'arch': args.arch,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : self.optimizer.state_dict(),
            }, is_best)

    def train(self, epoch):
        self.model.train()
        t = tqdm(self.train_loader)
        for batch_idx, (data, target) in enumerate(t):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            self.loss.append(loss.cpu().data.numpy())
            
            loss.backward()
            self.optimizer.step()
            
            t.set_postfix(loss=loss.data[0])

    def test(self):
        self.model.eval()
        test_loss = 0
        # correct = 0
        # psnr = 0.0
        for data, target in self.test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += self.loss_fn(output, target).data[0] # sum up batch loss
            #psnr = 10 * log10(1 / test_loss.data[0])
            #avg_psnr += psnr

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {0}'.format(test_loss))
        self.loss_val.append(test_loss)

optical_flow_network = optical_flow(32)
optical_flow_network.optimize(5, lr=1e-4)