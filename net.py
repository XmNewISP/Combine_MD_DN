#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

#简易动检网络
class MdSimpleNet(nn.Module):    
    def __init__(self):
        """Initializes U-Net."""
        super(MdSimpleNet, self).__init__()

    def forward(self,xset,pool_kernel,pool_stride,threshold):
        print("MdSimpleNet forward--->xset.size",xset.size())
        #滑窗求均值
        xset = nn.AvgPool2d(pool_kernel,pool_stride)(xset)
        #按通道进行分割
        x_list = torch.split(xset,1,dim=1)
        #求两帧的绝对差
        diffs = None
        x = x_list[0]
        for x_other in x_list[1:] :
            if diffs is None:
                diffs = torch.abs(x-x_other)
            else:
                diff = torch.abs(x-x_other)
                diffs = torch.cat((diffs,diff),1)
        print("MdSimpleNet forward--->diffs.size",diffs.size())
        #按通道求最值
        max_diff,_ = torch.max(diffs,1)
        print("MdSimpleNet forward--->max_diff.size",max_diff.size())
        #运动判断
        out = torch.ge(max_diff, threshold)
        print("MdSimpleNet forward--->out.size",out.size())
        return out.float()

class MdNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""
    
    def __init__(self, in_channels=6, out_channels=2):
        """Initializes U-Net."""

        super(MdNet, self).__init__()
        #
        self.net = nn.Sequential(               #32
            nn.Conv2d(in_channels, 128, 3),      #30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    #15
            nn.Conv2d(128, 256, 2),               #14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    #7
            nn.Conv2d(256, 512, 2),              #6
            nn.ReLU(inplace=True),      
            nn.MaxPool2d(2),                    #3
            nn.Conv2d(512, 1024, 2),             #2
            nn.ReLU(inplace=True),              
            nn.Conv2d(1024, out_channels, 2))    #1
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        return self.net(x)

#
class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            ################## 
            #nn.LeakyReLU(0.1)
            nn.Sigmoid())
        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)
