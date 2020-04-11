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

from net import MdSimpleNet,UNet
use_cuda = torch.cuda.is_available()

########################################################
#           可配参数
########################################################
#动检部分
MD_NET_RECEPTIVE_FIELD = 8      #动检窗口大小
MD_NET_STRIDE = 8               #动检滑窗步长
MD_NET_THRESHOULD = 0.09        #动静判断阈值(差值/256)
MD_HISTORY_NUM = 10             #动检判断用历史帧数(慢速移动物体)
#降噪部分
DENOISE_N2N_STRENGTH = 1.0      #n2n网络降噪强度(输出是噪声残差)
STATIC_DENOISE_WEMA = 0.9       #静止部分,时域衰减系数(WEMA)
########################################################

#降噪中间状态
denoise_status_image = None

#调试标志
DEBUG_DEFINE = 1

def parse_args():
    parser = ArgumentParser(description='PyTorch implementation of md & n2n (2018)')
    parser.add_argument('--data', help='dataset root path', default='../data')
    parser.add_argument('--result', help='dataset result path', default='../result')
    parser.add_argument('--n2n-ckpt', help='load md model checkpoint')    
    return parser.parse_args()

#https://blog.csdn.net/weixin_39128119/article/details/84172385
#对mask图像进行膨胀操作
def do_dilate(mask):
    #Image转cv
    mask = np.array(mask)
    #图像的腐蚀-膨胀
    mask = cv2.dilate(mask,None,iterations=5)
    #cv转Image
    return Image.fromarray(mask)

#输入图片,返回mask
def do_movedetect(model,frameA,frameB,frameC,frameD,frameE):
    tA = tvF.to_tensor(frameA).unsqueeze(0)
    tB = tvF.to_tensor(frameB).unsqueeze(0)
    tC = tvF.to_tensor(frameC).unsqueeze(0)
    tD = tvF.to_tensor(frameD).unsqueeze(0)
    tE = tvF.to_tensor(frameE).unsqueeze(0)
    if use_cuda:
        tA = tA.cuda()
        tB = tB.cuda()
        tC = tC.cuda()
        tD = tD.cuda()
        tE = tE.cuda()
    output = model(tA,tB,tC,tD,tE,MD_NET_RECEPTIVE_FIELD,MD_NET_STRIDE,MD_NET_THRESHOULD) #误差门限(/256)
    result = output.detach().squeeze(0).cpu()
    print("--->result.size:",result.size())
    md_fmap = tvF.to_pil_image(result)
    print("--->md_fmap.size:",md_fmap.size)
    
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
    md_mask  = Image.new("L",(frameE.size))
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
    denoise_status_image = denoise_status_image*0.80 + (1-0.80)*static_mask*noiseFrame
    denoise_status_image = denoise_status_image*static_mask + dynmic_mask*denoisedFrame
    #转Image
    out = denoise_status_image.astype(np.uint8)
    print("do_combine_fast-------->finish")
    return tvF.to_pil_image(out)
    
if __name__ == '__main__':
    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    md_simple = MdSimpleNet()
    n2n_model = UNet()
    if use_cuda:
        md_simple = md_simple.cuda()
        n2n_model = n2n_model.cuda()
    if use_cuda:
        n2n_model.load_state_dict(torch.load(params.n2n_ckpt))
    else:
        n2n_model.load_state_dict(torch.load(params.n2n_ckpt, map_location='cpu'))
    n2n_model.train(False)
    
    #处理每一张图片
    save_path = os.path.dirname(params.result)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    input_path = params.data
    namelist = [name for name in os.listdir(input_path) if "groundtruth" not in name]
    namelist.sort(key=lambda x:int(x.replace(".jpg","")))
    print(namelist)
    
    #5帧求最大值
    for a,b,c,d,e in zip(range(0,len(namelist)-4),range(1,len(namelist)-3),range(2,len(namelist)-2),range(3,len(namelist)-1),range(4,len(namelist))):
        print(a,b,c,d,e)
        print(namelist[a],namelist[b],namelist[c],namelist[d],namelist[e])
        imgA_path = os.path.join(input_path,namelist[a])
        imgB_path = os.path.join(input_path,namelist[b])
        imgC_path = os.path.join(input_path,namelist[c])
        imgD_path = os.path.join(input_path,namelist[d])
        imgE_path = os.path.join(input_path,namelist[e])
        imgA = Image.open(imgA_path).convert('RGB')
        imgB = Image.open(imgB_path).convert('RGB')
        imgC = Image.open(imgC_path).convert('RGB')
        imgD = Image.open(imgD_path).convert('RGB')
        imgE = Image.open(imgE_path).convert('RGB')
        #先做去噪
        dn_imgA = do_noise2noise(n2n_model,imgA)
        dn_imgB = do_noise2noise(n2n_model,imgB)
        dn_imgC = do_noise2noise(n2n_model,imgC)
        dn_imgD = do_noise2noise(n2n_model,imgD)
        dn_imgE = do_noise2noise(n2n_model,imgE)
        if DEBUG_DEFINE :  #调试信息输出
            dn_imgE.save(os.path.join(save_path, f'{namelist[e]}-1dn.jpg'))
        #再做动检
        md_mask = do_movedetect(md_simple,dn_imgA,dn_imgB,dn_imgC,dn_imgD,dn_imgE)
        if DEBUG_DEFINE :  #调试信息输出
            md_red_label = Image.new("RGB",(imgE.size),(255,0,0))          
            md_red = Image.composite(md_red_label,imgE,md_mask)
            md_red.save(os.path.join(save_path, f'{namelist[e]}-2md.jpg'))
 
        #对mask进行膨胀,填补空洞
        dilate_mask = do_dilate(md_mask)   
        if DEBUG_DEFINE :  #调试信息输出
            dilate_red = Image.composite(md_red_label,imgE,dilate_mask)
            dilate_red.save(os.path.join(save_path, f'{namelist[e]}-3dilate.jpg'))
   
        #再进行融合
        out2 = do_combine_fast(dilate_mask,imgE,dn_imgE)
        out2.save(os.path.join(save_path, f'ok_{namelist[e]}.jpg'))
        #exit(0)
