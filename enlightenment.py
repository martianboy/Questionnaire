import cv2,numpy as np
from math import cos,sin

def drawCollisionGraph(collisionMatrix, mode = True):
    M = collisionMatrix
    G = np.zeros((800,800), np.uint8)
    graphSize = M.shape[0]
    radius = 350
    dots = np.zeros((graphSize,2), np.int32)

    for i in range(graphSize):
        w = i * (2 * np.pi / graphSize)
        x = int(radius * cos(w)) + G.shape[0]/2
        y = int(radius * sin(w)) + G.shape[1]/2

        dots[i] = [x,y]

        cv2.circle(G, (x,y), 6, (255,255,255), -1)

    for i in range(graphSize):
        for j in range(i, graphSize):
            if mode != M[i,j]:
                cv2.line(G, (dots[i,0],dots[i,1]), (dots[j,0],dots[j,1]), (255,255,255), 2)

    return G

def HSVThreshold(im, hue, sat, val):
    hsvIm = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
    bH1 = (hsvIm[:,:,0] >= hue[0] / 1.4)
    bH2 = (hsvIm[:,:,0] <= hue[1] / 1.42)
    if hue[0] < hue[1]:
        bH = bH1 * bH2
    else:
        bH = bH1 + bH2

    bS1 = (hsvIm[:,:,1] >= sat[0] * 2.5)
    bS2 = (hsvIm[:,:,1] <= sat[1] * 2.6)
    if sat[0] < sat[1]:
        bS = bS1 * bS2
    else:
        bS = bS1 + bS2

    bV1 = (hsvIm[:,:,2] >= val[0] * 2.5)
    bV2 = (hsvIm[:,:,2] <= val[1] * 2.6)
    if val[0] < val[1]:
        bV = bV1 * bV2
    else:
        bV = bV1 + bV2
    
    thresh = np.array(bH * bS * bV, np.uint8) * 255

    return thresh

def getColorOfPoints(im, points):
    hsvIm = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
    hue = []
    sat = []
    val = []
    for p in points:
        block = hsvIm[p[1]-7:p[1]+7,p[0]-7:p[0]+7,:]
        hue.append(block[:,:,0].mean() * 1.411)
        sat.append(block[:,:,1].mean() / 2.55)
        val.append(block[:,:,2].mean() / 2.55)
    
    hue = np.array(hue)
    sat = np.array(sat)
    val = np.array(val)

    return ((hue.mean(),hue.std()),(sat.mean(),sat.std()),(val.mean(),val.std()))

