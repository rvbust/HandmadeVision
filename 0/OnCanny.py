#!/usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
from ImgCanvas import ImgCanvas
import sys

def OnCanny():
    img = cv2.imread(sys.argv[1], 1)
    edges = cv2.Canny(img,100,200)
    ys, xs = np.nonzero(edges)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    c = ImgCanvas(img)
    c.Point(xs, ys, size = 4, linewidth = 2)
    c.Show()

if __name__ == '__main__':
    OnCanny()
