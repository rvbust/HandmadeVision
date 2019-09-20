#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys 
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1)

class ImgCanvas(object):
    def __init__(self, imgornameorsize):
        if isinstance(imgornameorsize, str):            
            img = cv2.imread(imgornameorsize, 1)
        elif isinstance(imgornameorsize, (list, tuple)):
            w, h = imgornameorsize
            img = np.ones((h, w, 3), dtype = np.uint8) * 255
        else:
            img = imgornameorsize
        plt.imshow(img)

    def Plot(self, *args, scalex=True, scaley=True, data=None, **kwargs):
        plt.plot(*args, scalex=scalex, scaley=scaley, data=data, **kwargs)
        
    def Point(self, x, y, size = 1, color = (1,0,0), **kwargs): 
        if isinstance(x, (list, tuple, np.ndarray)):
            xs = x
            ys = y
        else:
            xs = [x]
            ys = [y]            
        plt.scatter(xs, ys, marker='.', s = size, color=color, **kwargs)
        
    def Line(self, a, b, color = (1,0,0), linewidth=1):
        if isinstance(a[0], (list, tuple)):
            la = a
            lb = b
        else:
            la = [a]
            lb = [b]
        for pa, pb in zip(la, lb):
            line = plt.Line2D(pa, pb, lw = linewidth, color = color)
            plt.gca().add_line(line)
        
    def Circle(self, center, radius, color = (1,0,0, 0.5), fill = True):
        circle = plt.Circle(center, radius = radius, color = color, fill = fill)
        plt.gca().add_patch(circle)
        
    def Rectangle(self, xy, width, height, color = (1,0,0,0.5), fill = True):
        rect = plt.Rectangle(xy, width, height, color = color, fill = fill)
        plt.gca().add_patch(rect)

    def Arrow(self, x, y, dx, dy, head_width=3, head_length=8, **kwargs):
        plt.arrow(x, y, dx, dy, head_width=head_width, head_length=head_length, **kwargs)
        
    def Ellipse(self, xy, width, height, angle = 0, **kwargs):
        ell = patches.Ellipse(xy, width, height, angle=angle, **kwargs)
        plt.gca().add_patch(ell)
    
    def Polygon(self, points, closed = True, fill = False, color = (1.0,0,0)):
        poly = plt.Polygon(points, closed = closed, fill = fill, edgecolor = color)
        plt.gca().add_patch(poly)
        
    def Show(self):
        plt.show()

def Test():
    import sys 
    c = ImgCanvas([800, 800])
    c.Plot(np.arange(300), 30 + 10*np.sin(np.arange(300)), linewidth=1, color = (1,0,0,0.5))
    c.Point(range(0, 200, 10), range(0, 200, 10))
    c.Line([(100, 100), (200, 100)], [(100, 200.4), (200, 300)])
    c.Line((200, 300), (210, 400))
    c.Circle((100, 100), 30, color = (1.0, 0, 0, 0.5))
    c.Rectangle((150,200), width = 30, height = 50, color = (0,1.0,0,0.5))
    c.Polygon([(100, 100), (300, 90), (400, 300), (88, 99)], color = (0,1,0))
    c.Ellipse((200, 200), width = 40, height = 30, angle = 30, color = (1.0,0,0), fill=False)
    c.Arrow(10, 10, 300, 130, head_width=3, head_length=8, color = (1.0,0,0, 0.5))
    c.Show();

if __name__ == '__main__':
    # Test()
    c = ImgCanvas(sys.argv[1])
    c.Show()
    
