import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, upsample=False, transpose=False):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.transpose = transpose

        if (upsample):
            self.upsample = nn.Upsample(scale_factor=2)
            if (transpose):
                self.conv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, output_padding=1)
            else:
                self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1)
        else:
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride)

        self.reflection = nn.ReflectionPad2d(1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        
        if (self.upsample):
            out = self.upsample(out)

        out = self.reflection(out)
        out = self.conv(out)
            
        return out
    
class network_optical_flow(nn.Module):
    def __init__(self):
        super(network_optical_flow, self).__init__()
        self.A01 = ConvBlock(2, 64)
        
        self.C11 = ConvBlock(64, 64, stride=2)
        self.C12 = ConvBlock(64, 64)
        self.C13 = ConvBlock(64, 64)
        self.C14 = ConvBlock(64, 64)
        
        self.C21 = ConvBlock(64, 64)
        self.C22 = ConvBlock(64, 64)
        self.C23 = ConvBlock(64, 64)
        self.C24 = ConvBlock(64, 64)
        
        self.C31 = ConvBlock(64, 128, stride=2)
        self.C32 = ConvBlock(128, 128)
        self.C33 = ConvBlock(128, 128)
        self.C34 = ConvBlock(128, 128)
        
        self.C41 = ConvBlock(128, 256, stride=2)
        self.C42 = ConvBlock(256, 256)
        self.C43 = ConvBlock(256, 256)
        self.C44 = ConvBlock(256, 256)
        
        self.C51 = ConvBlock(256, 128, stride=2, upsample=True)
        self.C52 = ConvBlock(128, 128)
        self.C53 = ConvBlock(128, 128)
        self.C54 = ConvBlock(128, 128)
        
        self.C61 = ConvBlock(128, 64, stride=2, upsample=True)
        self.C62 = ConvBlock(64, 64)
        self.C63 = ConvBlock(64, 64)
        self.C64 = ConvBlock(64, 64)
        
        self.C71 = ConvBlock(64, 64, stride=2, upsample=True)

        self.C72 = nn.Conv2d(64, 2, kernel_size=1)
        
    def forward(self, x):
        A01 = self.A01(x)
        
        C11 = self.C11(A01)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = self.C14(C13)
        C14 += C11
        
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)
        C24 = self.C24(C23)
        C24 += C21
        
        C31 = self.C31(C24)
        C32 = self.C32(C31)
        C33 = self.C33(C32)
        C34 = self.C34(C33)
        C34 += C31
        
        C41 = self.C41(C34)
        C42 = self.C42(C41)
        C43 = self.C43(C42)
        C44 = self.C44(C43)
        C44 += C41
        
        C51 = C34 + self.C51(C44)
        C52 = self.C52(C51)
        C53 = self.C53(C52)
        C54 = self.C54(C53)
        C54 += C51
        
        C61 = C24 + self.C61(C54)
        C62 = self.C62(C61)
        C63 = self.C63(C62)
        C64 = self.C64(C63)
        C64 += C61
        
        C71 = self.C71(C64)
        out = self.C72(C71)
        
        return out
    
class network_spatial_transformer(nn.Module):
    def __init__(self):
        super(network_spatial_transformer, self).__init__()
        
    def forward(self, x, flow):
        flow_reshape = flow.transpose(1,2).transpose(2,3)
        
        out = torch.nn.functional.grid_sample(x, flow_reshape)
        
        return out
    
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.layer1 = network_optical_flow()
        self.layer2 = network_spatial_transformer()
        
    def forward(self, x):
        
        flow = self.layer1(x)
        out = self.layer2(x[:,0:1,:,:], flow)
        
        return out

class network_syn(nn.Module):
    def __init__(self):
        super(network_syn, self).__init__()
        self.layer1 = network_optical_flow()
        self.layer2 = network_spatial_transformer()
        
    def forward(self, x):
        
        flow = self.layer1(x)
        out = self.layer2(x[:,0:1,:,:], flow)
        
        return out, flow