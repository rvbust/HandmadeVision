#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def RandCVColor():
    return np.random.randint(30, 220, size = 3)

def RandCVColors(ncolor):
    return np.random.randint(30, 220, size = 3 * ncolor).reshape(-1, 3)

def RandColor():
    return 0.2 + 0.8 * np.random.rand(3)

def RandColors(ncolor):
    return 0.2 + 0.8 * np.random.rand(ncolor, 3)

def MakeImage(w, h, val = 255):
    img = np.ones((h, w), dtype = np.uint8) * val
    return img

def MakeBWImage(imgsize = (200, 200)):
    """ Make a white-black image for edge detection stuff 
    """
    w,h = imgsize
    img = MakeImage(w, h)
    img[:, :w//2] = 0
    return img

def MakeEdgeImage(imgsize = (200, 200), edgesize = 1):
    """ make an image with an edge in the middle 
    """
    w, h = imgsize
    img = MakeImage(w, h)
    mid = w//2
    img[:, mid:mid+edgesize] = 0
    return img 

def MakeRect(lo, hi):
    """ Generate a random rectangle with all points inside the [lo, hi] range 

    :param lo: all x <= lo
    :param hi: all y >= hi
    :returns: a rect represented in a numpy array with shape (4,)
    :rtype: np.ndarray 

    """
    x1 = np.random.randint(lo, hi+1)
    y1 = np.random.randint(lo, hi+1)
    x2 = np.random.randint(lo, hi+1)
    y2 = np.random.randint(lo, hi+1)
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    ymin = min(y1, y2)
    ymax = max(y1, y2)
    return np.array([x1, y1, xmax - xmin, ymax - ymin])

def RectVertices(rect):
    x,y,w,h = rect
    return np.array([[x,y],
                     [x+w, y],
                     [x+w, y+h],
                     [x, y+h]], dtype = np.float)

def Rotate2D(pts, center, theta):
    """ Rotate points around a center 

    :param pts: the pts need to be rotated, should be np.ndarray 
    :param center: center point to rotate around 
    :param theta: angle in degree 
    :returns: return the same type as pts but rotated 
    :rtype: np.nparray 
    
    
    """
    from numpy import sin, cos
    rad = np.deg2rad(theta)
    R = np.array([[cos(rad), -sin(rad)],
                  [sin(rad), cos(rad)]])
    tpts = pts.copy() - center
    return np.dot(R, tpts.T).T + center

def MakeKPImage():
    img = MakeImage(w = 800, h = 600)
    # draw a black filled rectanle 
    img[40:100, 100:240] = 0
    
    # draw a rectangle with linewidth == 1
    img[140:200, 100:240] = 0
    img[141:199, 101:239] = 255

    # draw a rectangle with linewidth == 1
    img[240:300, 100:240] = 0
    img[243:297, 103:237] = 255

    # a rotated rectangle
    pts = np.array([[100, 340],
                    [100, 400],
                    [240, 400],
                    [240, 340]], dtype = np.float32)
    
    rpts = Rotate2D(pts, pts.mean(axis = 0), 30)

    ipts = np.round(rpts).astype(np.int32)
    
    cv2.line(img, tuple(ipts[0]), tuple(ipts[1]), color = (0,0,0), thickness=1, lineType = cv2.LINE_AA)
    cv2.line(img, tuple(ipts[1]), tuple(ipts[2]), color = (0,0,0), thickness=1, lineType = cv2.LINE_AA)
    cv2.line(img, tuple(ipts[2]), tuple(ipts[3]), color = (0,0,0), thickness=1, lineType = cv2.LINE_AA)
    cv2.line(img, tuple(ipts[3]), tuple(ipts[0]), color = (0,0,0), thickness=1, lineType = cv2.LINE_AA)
    
    # draw a circle with anti-aliasing
    cv2.circle(img, (300, 80), 11, color = (0,0,0), thickness = 1, lineType = cv2.LINE_AA)
    cv2.circle(img, (400, 80), 11, color = (0,0,0), thickness = 2, lineType = cv2.LINE_AA)
    cv2.circle(img, (500, 80), 11, color = (0,0,0), thickness = 3, lineType = cv2.LINE_AA)
    cv2.circle(img, (600, 80), 11, color = (0,0,0), thickness = 4, lineType = cv2.LINE_AA)
    cv2.circle(img, (700, 80), 11, color = (0,0,0), thickness = -1, lineType = cv2.LINE_AA)

    cv2.circle(img, (400, 180), 40, color = (0,0,0), thickness = 1, lineType = cv2.LINE_AA)
    
    cv2.circle(img, (540, 180), 40, color = (0,0,0), thickness = 3, lineType = cv2.LINE_AA)

    # draw a cross with line-size 1
    img[300:500, 400] = 0; img[400, 300:500] = 0

    # draw a cross by rotated lines
    cv2.line(img, (500, 300), (550, 460), color = (0,0,0), thickness = 1, lineType = cv2.LINE_AA)
    cv2.line(img, (500, 460), (550, 300), color = (0,0,0), thickness = 1, lineType = cv2.LINE_AA)
    
    return img 

def MakeRotatingRectImage(degree):
    img = MakeImage(300, 300)
    # a rotated rectangle
    pts = np.array([[100, 80],
                    [100, 200],
                    [200, 200],
                    [200, 80]], dtype = np.float32)    
    rpts = Rotate2D(pts, pts.mean(axis = 0), degree)
    ipts = np.round(rpts).astype(np.int32)
    cv2.fillConvexPoly(img, ipts, color = (0,0,0), lineType = cv2.LINE_AA)
    return img 

if __name__ == '__main__':
    img = MakeKPImage()
    cv2.imwrite("../Data/kp.png", img)    
    cv2.imshow("img", img)
    cv2.waitKey(0)

    inv_img = 255 - img
    cv2.imshow("inv_img", inv_img)
    cv2.waitKey(0)
    cv2.imwrite("../Data/kp_inv.png", inv_img)    
