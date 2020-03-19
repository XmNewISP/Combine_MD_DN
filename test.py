#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import numpy as np
from argparse import ArgumentParser

from ..movedetect-pytorch.src.unet import MdNet
from ..noise2noise-pytorch.src.unet import UNet
use_cuda = torch.cuda.is_available()

#必须和MdNet的参数保持一致
MD_NET_STRIDE = 8
MD_NET_RECEPTIVE_FIELD = 32

def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of md & n2n (2018)')

    # Data parameters
    parser.add_argument('--data', help='dataset root path', default='../data')
    parser.add_argument('--result', help='dataset result path', default='../result')
    parser.add_argument('--md-ckpt', help='load md model checkpoint')
    parser.add_argument('--n2n-ckpt', help='load md model checkpoint')    
    return parser.parse_args()

#输入图片,返回mask
def do_movedetect(model,oldframe,newframe):
    oldT = tvF.to_tensor(oldframe).unsqueeze(0)
    newT = tvF.to_tensor(newframe).unsqueeze(0)
    if use_cuda:
        oldT = oldT.cuda()
        newT = newT.cuda()
    input = torch.cat((oldT,newT),dim=1)
    output = model(input).detach()
    result = F.softmax(output, dim=1).squeeze(0).detach().cpu()
    md_fmap = tvF.to_pil_image(result[1])
    #放大到原图:有技巧《学习笔记:关于感受野》
    image_box_x1 = MD_NET_RECEPTIVE_FIELD//2
    image_box_y1 = MD_NET_RECEPTIVE_FIELD//2
    image_box_x2 = (md_fmap.size[0]-1)*MD_NET_STRIDE+MD_NET_RECEPTIVE_FIELD//2
    image_box_y2 = (md_fmap.size[1]-1)*MD_NET_STRIDE+MD_NET_RECEPTIVE_FIELD//2
    #只是中间有效部分
    rsz_w = image_box_x2-image_box_x1+1
    rsz_h = image_box_y2-image_box_y1+1
    md_rsz = md_fmap.resize((rsz_w,rsz_h))
    #补全四周，得到原图的mask
    md_mask  = Image.new("L",(img.size))
    md_mask.paste(md_rsz,(image_box_x1,image_box_y1))
    return md_mask
    
def do_noise2noise(model,frame):
    #Unet的设计导致要32对齐
    w, h = frame.size   
    if w % 32 != 0:
        w = (w//32)*32
    if h % 32 != 0:
        h = (h//32)*32
    crop_img = tvF.crop(frame, 0, 0, h, w)
    source = tvF.to_tensor(crop_img)
    source = source.unsqueeze(0)
    if use_cuda:
        source = source.cuda()
    # Denoise
    denoised = model(source).detach()
    denoised = denoised.cpu()
    denoised = denoised.squeeze(0)
    denoised = tvF.to_pil_image(denoised)
    #贴合原图大小
    frame.paste(denoised,(0, 0))
    return frame

#降噪中间状态
MD_THRESHOLD[3] = [32,64,96]          #
TIME_FILTER[3] = [0.99,0.95,0.90]     #0.95相当于历史20帧的加权滑动平均
denoise_status_image = None 
def do_combine(mdMask,noiseFrame,denoisedFrame):
    global denoise_status_image
    if denoise_status_image == None:
        print(mdMask.size)
        denoise_status_image = np.array(denoisedFrame)
        print(denoise_status_image.shape)
    
    #根据动检结果进行融合
    w,h = mdMask.size
    for i in range(h):
        for j in range(w):
            mdScore = mdMask.getpixel((i,j))
            if(mdScore<MD_THRESHOLD[0]):  #静止_0
                val = noiseFrame.getpixel((i,j)) #RGB
                denoise_status_image[i,j] = \
                denoise_status_image[i,j]*TIME_FILTER[0] + val*(1-TIME_FILTER[0])
            elif(mdScore<MD_THRESHOLD[1]):  #静止_1
                val = noiseFrame.getpixel((i,j)) #RGB
                denoise_status_image[i,j] = \
                denoise_status_image[i,j]*TIME_FILTER[1] + val*(1-TIME_FILTER[1])
            elif(mdScore<MD_THRESHOLD[2]):  #静止_2
                val = noiseFrame.getpixel((i,j)) #RGB
                denoise_status_image[i,j] = \
                denoise_status_image[i,j]*TIME_FILTER[2] + val*(1-TIME_FILTER[2])
            else:   #运动,以2D降噪为准
                val = denoisedFrame.getpixel((i,j))
                denoise_status_image[i,j] = val
    denoise_status_image = np.array(denoisedFrame)
    print("denoise_status_image.dtype=",denoise_status_image.dtype)
    out = denoise_status_image.astype(np.uint8)
    return tvF.to_pil_image(out)

if __name__ == '__main__':
    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    md_model = MdNet()
    n2n_model = UNet()
    if use_cuda:
        md_model = md_model.cuda()
        n2n_model = n2n_model.cuda()
    if use_cuda:
        md_model.load_state_dict(torch.load(params.md_ckpt))
        n2n_model.load_state_dict(torch.load(params.n2n_ckpt))
    else:
        md_model.load_state_dict(torch.load(params.md_ckpt, map_location='cpu'))
        n2n_model.load_state_dict(torch.load(params.n2n_ckpt, map_location='cpu'))
    n2n_model.train(False)

    #处理每一张图片
    save_path = os.path.dirname(self.p.result)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    input_path = self.p.data
    namelist = os.listdir(input_path)
    namelist.sort()
    print(namelist)
    img_path = os.path.join(input_path,name)
    for a,b in zip(range(0,len(namelist)-1),range(1,len(namelist))):
        print(a,b)
        imgA_path = os.path.join(input_path,namelist[a])
        imgB_path = os.path.join(input_path,namelist[b])
        imgA = Image.open(imgA_path).convert('RGB')
        imgB = Image.open(imgB_path).convert('RGB')
        print(imgA_path,imgB_path)
        print(imgA.size)
        assert(imgA.size == imgB.size)
        #先做动检
        md_mask = do_movedetect(md_model,imgA,imgB)
        #再做去噪
        dn_imgB = do_noise2noise(n2n_model,imgB)
        #再进行融合
        out = do_combine(md_mask,imgB,dn_imgB)
        #print(md_red.size)
        fname = os.path.basename(imgB_path)
        out.save(os.path.join(save_path, f'{fname}_dn.jpg'))
