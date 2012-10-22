import cv2, csv, numpy as np

from os import chdir
from cv2 import imread, imwrite, cvtColor
from scipy.cluster import hierarchy as hier

from paper import FindPapers

def main():
  im = cv2.imread('test/a3.jpg')
  papers = FindPapers(im)
  
  h = np.array([[0,0],[815,0],[815,1064],[0,1064]], dtype=np.float32)
  for i,paper in enumerate(papers):
    perspective_matrix = cv2.getPerspectiveTransform(paper, h)
    warp = cv2.warpPerspective(im, perspective_matrix, (815,1064))
    imwrite("result/" + str(i) + ".jpg", warp)    

if __name__ == '__main__':
  main()
