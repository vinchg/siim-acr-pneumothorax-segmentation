


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, unet_block):
        super(ResidualBlock, self).__init__()
        self.block = unet_block(in_channels,out_channels)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


def unet_block(chanIn, chanOut, ks=1, stride=1):
    return nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(chanOut, chanOut, kernel_size=3,stride=stride, padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(inplace=True)
            )

def unet_block_large(chanIn, chanOut, ks=1, stride=1):
    return nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Conv2d(chanOut, chanOut, kernel_size=3,stride=stride, padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(),
            nn.Conv2d(chanOut, chanOut, kernel_size=3,stride=stride, padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(),
            # nn.Dropout(.2),
            # nn.Conv2d(chanOut, chanOut, kernel_size=3,stride=stride, padding=1),
            # nn.BatchNorm2d(chanOut),
            # nn.ReLU()
            )

def unet_block_3(chanIn, chanOut, ks=1, stride=1):
    return nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Conv2d(chanOut, chanOut, kernel_size=3,stride=stride, padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU(),
            nn.Conv2d(chanOut, chanOut, kernel_size=3,stride=stride, padding=1),
            nn.BatchNorm2d(chanOut),
            nn.ReLU()
            )
   
def conv3x3(in_channels, out_channels, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


#THICC NET
class Thicc_Net(nn.Module):
    def __init__(self, layers,dila=1, dilb=2, dilc=3):
        super(Thicc_Net, self).__init__()
        self.chn = 16
       
        self.block1 = self.make_layer(1,self.chn, layers[0], unet_block_large)
        self.pool_1 = nn.MaxPool2d(2)
        self.block2 = self.make_layer(self.chn,self.chn*2, layers[1], unet_block_large)
        self.pool_2 = nn.MaxPool2d(2)
        self.block3 = self.make_layer(self.chn*2,self.chn*4, layers[2], unet_block_large)
        self.pool_3 = nn.MaxPool2d(2)
        
        self.block4 = self.make_layer(self.chn*4,self.chn*8, layers[2], unet_block_large)
        self.pool_4 = nn.MaxPool2d(2)
        self.bottom_layer = self.make_layer(self.chn*8,self.chn*16, layers[3], unet_block_large)
       
        self.up1_1x1 = nn.Sequential( 
                        nn.Upsample(scale_factor=2,mode='bilinear'),
                        nn.Conv2d(self.chn*16, self.chn*8, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.upconv1 = unet_block(self.chn*16, self.chn*8)

        self.up2_1x1 = nn.Sequential( 
                        nn.Upsample(scale_factor=2,mode='bilinear'),
                        nn.Conv2d(self.chn*8, self.chn*4, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.upconv2 = unet_block(self.chn*8, self.chn*4)

        self.up3_1x1 = nn.Sequential( 
                        nn.Upsample(scale_factor=2,mode='bilinear'),
                        nn.Conv2d(self.chn*4, self.chn*2, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.upconv3 = unet_block(self.chn*4, self.chn*2)
    
        self.up4_1x1 = nn.Sequential( 
                        nn.PixelShuffle(2),
                        nn.Conv2d((self.chn*2)//4, self.chn, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.upconv4 = unet_block(self.chn*2, self.chn)
        self.convL = conv3x3(self.chn,1)

    def make_layer(self, in_channels, out_channels, blocks, unet_block):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels,unet_block))  
        return nn.Sequential(*layers)

    def forward(self, x):
        skip1 = self.block1(x) 
        pool_1 = self.pool_1(skip1)      
        skip2 = self.block2(pool_1)            
        pool_2 = self.pool_2(skip2) 
        
        skip3 = self.block3(pool_2)            
        pool_3 = self.pool_3(skip3) 

        skip4 = self.block4(pool_3)            
        pool_4 = self.pool_4(skip4) 
        
        bottom = self.bottom_layer(pool_4) 

        up1 = self.up1_1x1(bottom)    
        cat1 = torch.cat([skip4,up1],dim=1) 
        out1 = self.upconv1(cat1)
        up2 = self.up2_1x1(out1)    
        cat2 = torch.cat([skip3,up2],dim=1) 
        out2 = self.upconv2(cat2)
        up3 = self.up3_1x1(out2)    
        cat3 = torch.cat([skip2,up3],dim=1) 
        out3 = self.upconv3(cat3)
        up4 = self.up4_1x1(out3)    
        cat4 = torch.cat([skip1,up4],dim=1) 
        out4 = self.upconv4(cat4)
        x = self.convL(out4) 
        x = F.sigmoid(x)
        return x
    


class Thicc_5Pool(nn.Module):
    def __init__(self, layers,dila=1, dilb=2, dilc=3):
        super(Thicc_5Pool, self).__init__()
        self.chn = 16
       
        self.block1 = self.make_layer(1,self.chn, layers[0], unet_block_large)
        self.pool_1 = nn.MaxPool2d(2)
        self.block2 = self.make_layer(self.chn,self.chn*2, layers[1], unet_block_large)
        self.pool_2 = nn.MaxPool2d(2)
        self.block3 = self.make_layer(self.chn*2,self.chn*4, layers[2], unet_block_large)
        self.pool_3 = nn.MaxPool2d(2)
        
        self.block4 = self.make_layer(self.chn*4,self.chn*8, layers[2], unet_block_large)
        self.pool_4 = nn.MaxPool2d(2)
        self.bottom_layer = self.make_layer(self.chn*8,self.chn*16, layers[3], unet_block_large)
       
        self.up1_1x1 = nn.Sequential( 
                        nn.Upsample(scale_factor=2,mode='bilinear'),
                        nn.Conv2d(self.chn*16, self.chn*8, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.upconv1 = unet_block(self.chn*16, self.chn*8)

        self.up2_1x1 = nn.Sequential( 
                        nn.Upsample(scale_factor=2,mode='bilinear'),
                        nn.Conv2d(self.chn*8, self.chn*4, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.upconv2 = unet_block(self.chn*8, self.chn*4)

        self.up3_1x1 = nn.Sequential( 
                        nn.Upsample(scale_factor=2,mode='bilinear'),
                        nn.Conv2d(self.chn*4, self.chn*2, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.upconv3 = unet_block(self.chn*4, self.chn*2)
    
        self.up4_1x1 = nn.Sequential( 
                        nn.PixelShuffle(2),
                        nn.Conv2d((self.chn*2)//4, self.chn, kernel_size=1),
                        nn.ReLU(inplace=True)
                        )
        self.upconv4 = unet_block(self.chn*2, self.chn)
        self.convL = conv3x3(self.chn,1)

    def make_layer(self, in_channels, out_channels, blocks, unet_block):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels,unet_block))  
        return nn.Sequential(*layers)

    def forward(self, x):
        skip1 = self.block1(x) 
        pool_1 = self.pool_1(skip1)      
        skip2 = self.block2(pool_1)            
        pool_2 = self.pool_2(skip2) 
        
        skip3 = self.block3(pool_2)            
        pool_3 = self.pool_3(skip3) 

        skip4 = self.block4(pool_3)            
        pool_4 = self.pool_4(skip4) 
        
        bottom = self.bottom_layer(pool_4) 

        up1 = self.up1_1x1(bottom)    
        cat1 = torch.cat([skip4,up1],dim=1) 
        out1 = self.upconv1(cat1)
        up2 = self.up2_1x1(out1)    
        cat2 = torch.cat([skip3,up2],dim=1) 
        out2 = self.upconv2(cat2)
        up3 = self.up3_1x1(out2)    
        cat3 = torch.cat([skip2,up3],dim=1) 
        out3 = self.upconv3(cat3)
        up4 = self.up4_1x1(out3)    
        cat4 = torch.cat([skip1,up4],dim=1) 
        out4 = self.upconv4(cat4)
        x = self.convL(out4) 
        return x

