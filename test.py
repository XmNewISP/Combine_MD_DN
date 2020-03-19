#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import numpy as np
import time
from PIL import Image
import cv2
from argparse import ArgumentParser

from net import MdNet,UNet
use_cuda = torch.cuda.is_available()

#必须和MdNet的参数保持一致
MD_NET_STRIDE = 8
MD_NET_RECEPTIVE_FIELD = 32

#降噪中间状态
denoise_status_image = None

#调试标志
DEBUG_DEFINE = 1

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

#https://blog.csdn.net/weixin_39128119/article/details/84172385
#对mask图像进行膨胀操作
def do_dilate(mask):
    #Image转cv
    mask = np.array(mask)
    #设置卷积核5*5
    kernel = np.ones((8,8),np.uint8)
    #图像的膨胀-膨胀
    mask = cv2.dilate(mask,None,iterations=5)
    #cv转Image
    return Image.fromarray(mask) 

#输入图片,返回mask
def do_movedetect(model,oldframe,curframe):
    oldT = tvF.to_tensor(oldframe).unsqueeze(0)
    newT = tvF.to_tensor(curframe).unsqueeze(0)
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
    md_mask  = Image.new("L",(curframe.size))
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

def do_combine(mdMask,noiseFrame,denoisedFrame):
    print("do_combine-------->")
    global denoise_status_image
    if denoise_status_image is None:
        print(mdMask.size)
        denoise_status_image = np.array(denoisedFrame,dtype=np.float32)
        print(denoise_status_image.shape)   #HWC
        print(denoise_status_image.dtype)
    
    #根据动检结果进行融合
    width,height = mdMask.size
    for y in range(height):
        for x in range(width):
            mdScore = mdMask.getpixel((x,y))
            if(mdScore<=64):  #静止_0
                val = noiseFrame.getpixel((x,y)) #RGB
                denoise_status_image[y,x] = \
                denoise_status_image[y,x]*0.90 + np.array(val)*(1-0.90)
            else:   #运动,以2D降噪为准
                val = denoisedFrame.getpixel((x,y))
                denoise_status_image[y,x] = np.array(val)
    out = denoise_status_image.astype(np.uint8)
    print("do_combine-------->finish")
    return tvF.to_pil_image(out)
    
def do_combine_fast(mdMask,noiseFrame,denoisedFrame):
    print("do_combine_fast-------->")
    global denoise_status_image
    if denoise_status_image is None:
        print(mdMask.size)
        denoise_status_image = np.array(denoisedFrame,dtype=np.float32)
        print(denoise_status_image.shape)   #HWC
        print(denoise_status_image.dtype)
    #根据动检结果进行融合
    mask = np.array(mdMask)
    mask_bd = np.zeros_like(noiseFrame)
    print("mask_bd.dtype--->",mask_bd.dtype)
    mask_bd[:,:,0] = mask
    mask_bd[:,:,1] = mask
    mask_bd[:,:,2] = mask
    noiseFrame = np.array(noiseFrame)
    denoisedFrame = np.array(denoisedFrame)
    print("denoisedFrame.dtype--->",denoisedFrame.dtype)    
    #二值化
    static_mask = 1*(mask_bd <= 64)
    dynmic_mask = 1-static_mask
    #print(static_mask)            
    #print(dynmic_mask)        
    denoise_status_image = denoise_status_image*0.90 + (1-0.90)*static_mask*noiseFrame
    denoise_status_image = denoise_status_image*static_mask + dynmic_mask*denoisedFrame    
    #转Image
    out = denoise_status_image.astype(np.uint8)
    print("do_combine_fast-------->finish")
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
    save_path = os.path.dirname(params.result)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    input_path = params.data
    namelist = [name for name in os.listdir(input_path) if name!="groundtruth.jpg"]
    namelist.sort(key=lambda x:int(x.replace(".jpg","")))
    #print(namelist)
    
    for a,b in zip(range(0,len(namelist)-1),range(1,len(namelist))):
        print(a,b)
        print(namelist[a],namelist[b])
        imgA_path = os.path.join(input_path,namelist[a])
        imgB_path = os.path.join(input_path,namelist[b])
        imgA = Image.open(imgA_path).convert('RGB')
        imgB = Image.open(imgB_path).convert('RGB')
        print(imgA_path,imgB_path)
        print(imgA.size)
        assert(imgA.size == imgB.size)
        #先做动检
        md_mask = do_movedetect(md_model,imgA,imgB)
        if DEBUG_DEFINE :  #调试信息输出
            md_red_label = Image.new("RGB",(imgB.size),(255,0,0))          
            md_red = Image.composite(md_red_label,imgB,md_mask)
            md_red.save(os.path.join(save_path, f'{namelist[b]}-md.jpg'))
        #对mask进行膨胀,填补空洞
        dilate_mask = do_dilate(md_mask)   
        if DEBUG_DEFINE :  #调试信息输出
            dilate_red = Image.composite(md_red_label,imgB,dilate_mask)
            dilate_red.save(os.path.join(save_path, f'{namelist[b]}-dilate.jpg'))
        #再做去噪
        dn_imgB = do_noise2noise(n2n_model,imgB)
        if DEBUG_DEFINE :  #调试信息输出
            dn_imgB.save(os.path.join(save_path, f'{namelist[b]}-dn.jpg'))
        #再进行融合
        #out1 = do_combine(dilate_mask,imgB,dn_imgB)
        #out1.save(os.path.join(save_path, f'{namelist[b]}_ok1.jpg'))
        out2 = do_combine_fast(dilate_mask,imgB,dn_imgB)
        out2.save(os.path.join(save_path, f'{namelist[b]}_ok2.jpg'))
        #exit(0)
