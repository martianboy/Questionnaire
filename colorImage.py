import cv2, numpy as np
from scipy.cluster.hierarchy import fclusterdata
from utils import block

def normalize(im):
    mono = np.array(im, np.uint8)
    for i in range(3):
        fmono = np.array(im[:,:,i], np.float32)
        minValue = fmono.min()
        maxValue = fmono.max()

        mono[:,:,i] = np.array(255 * (fmono[:,:] - minValue)/(maxValue - minValue), np.uint8)
    return mono

def matchColor(color, criteria):
    for i,crit in enumerate(criteria):
        hue = crit[0]
        sat = crit[1]
        val = crit[2]

        bH1 = (color[0] >= hue[0] / 1.4)
        bH2 = (color[0] <= hue[1] / 1.42)
        if hue[0] < hue[1]:
            bH = bH1 * bH2
        else:
            bH = bH1 + bH2

        bS1 = (color[1] >= sat[0] * 2.5)
        bS2 = (color[1] <= sat[1] * 2.6)
        if sat[0] < sat[1]:
            bS = bS1 * bS2
        else:
            bS = bS1 + bS2

        bV1 = (color[2] >= val[0] * 2.5)
        bV2 = (color[2] <= val[1] * 2.6)
        if val[0] < val[1]:
            bV = bV1 * bV2
        else:
            bV = bV1 + bV2

        if bH * bS * bV: return i

    return -1

# FIXME: This function is very slow!
def searchForColorPoints(im, criteria):
    points = []
    pointColors = []
    hsvIm = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
    for i in range(11,im.shape[1]-11, 10):
        for j in range(11,im.shape[0]-11, 10):
            b = block(hsvIm, (i,j), 8)
            if b[:,:,0].std()>25: continue
            
            color = (b[:,:,0].mean(),b[:,:,1].mean(),b[:,:,2].mean())
            matchedColor = matchColor(color, criteria)
            if matchedColor >= 0:
                points.append((i,j))
                pointColors.append(matchedColor)

    points = np.array(points, np.float16)
    cluster = fclusterdata(points, 10, 'distance')

    centroids = []
    for i in range(len(criteria)):
        centroids.append([])
    
    for i in range(1,cluster.max() + 1):
        b = cluster == i
        
        c = np.zeros((1,2), np.int16);
        for p in points[b.argsort()[len(b)-sum(b):]]:
            c = c + p/sum(b)

        centroids[pointColors[b.argsort()[len(b)-sum(b)]]].append(c[0])
    
    return centroids
