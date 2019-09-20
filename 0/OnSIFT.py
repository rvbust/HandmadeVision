#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
from ImgCanvas import ImgCanvas
from Utils import RandColors, MakeRotatingRectImage

def OnSIFT(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp   = sift.detect(gray,None)
    
    xs = np.array([p.pt[0] for p in kp])
    ys = np.array([p.pt[1] for p in kp])
    
    c = ImgCanvas(img)
    c.Point(xs, ys, color = RandColors(ncolor = len(xs)), size = 9, linewidth = 3)
    c.Show()    


def OnSIFT_Batch():
    for i in range(0, 180, 10):
        img = MakeRotatingRectImage(i)
        OnSIFT(img)
        
if __name__ == '__main__':
    if len(sys.argv) == 2:
        img  = cv2.imread(sys.argv[1], 0)
        OnSIFT(img)
    else:
        OnSIFT_Batch()

