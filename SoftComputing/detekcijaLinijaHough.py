import numpy as np
import slika as sl
import cv2
import os

from scipy import ndimage
from matplotlib.pyplot import cm
import itertools
import time


# detekcija linija na slici pomocu Hjuove transformacije
def lineImage(img):

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # konverzija BGR u RGB 
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) # slika konvertovana u crno-belu
    ret, t = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY) # primena binarnog tresholda
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    distResolution = 1
    angleResolution = np.pi / 180 # da bi bilo u radijanima
    thresholdParam = 100
    maxGap = 100
    minLength = 100

    # primena Hjuove transformacije sa prosledjenim parametrima
    lines = cv2.HoughLinesP(image = edges, rho = distResolution, theta = angleResolution, threshold = thresholdParam, minLineLength = minLength, maxLineGap = maxGap)

    xDonje, yDonje, xGornje, yGornje = [], [], [], [] # donja leva, donja desna, gornja leva, gornja desna
    # ima 4 jer je linija deblja
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                xDonje.append(x1)
                yDonje.append(y2)
                xGornje.append(x2)
                yGornje.append(y1)
        xDonje1, yDonje1, xGornje1, yGornje1 = min(xDonje), min(yDonje), max(xGornje), max(yGornje)

        return xDonje1, yGornje1, xGornje1, yDonje1 # vraca koordinate


# ovu pozivam za pronalazak plave
def findBlueLine2(frame):
    lower = np.array([0, 0, 100], dtype="uint8")
    upper = np.array([50, 50, 255], dtype="uint8")
    blue = sl.in_rangeImage(frame, lower, upper)
    xmin1, ymax1, xmax1, ymin1 = lineImage(blue)
    blueLine = (xmin1, ymax1), (xmax1, ymin1)

    #cv2.imshow('blueLine', blue)   
    return blueLine


# ovu pozivam za pronalazak zelene
def findGreenLine2(frame):
    lower = np.array([0, 180, 0 ], dtype="uint8")
    upper = np.array([50, 255, 50], dtype="uint8")    
    green = sl.in_rangeImage(frame, lower, upper)
    xmin2, ymax2, xmax2, ymin2 = lineImage(green)
    greenLine = (xmin2, ymax2), (xmax2, ymin2)

    #cv2.imshow('greenLine', green)
    return greenLine



  