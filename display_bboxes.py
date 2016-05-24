import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import cv2
import numpy as np
import pickle
import imagePyramid as ip
import matplotlib.pyplot as plt
import math


def draw_bboxes(bboxes, image_path, scale):

    my_im = cv2.imread(image_path)
    #my_im = color.rgb2gray(my_im)
    for depth,rects in bboxes.iteritems():
        if depth == 1:
            for my_rect in rects:  
                cv2.rectangle(my_im,(my_rect[0],my_rect[1]),(my_rect[2],my_rect[3]),(255,0,0),1)
                plt.imshow(my_im)

        else:
            scale_back = math.pow(scale, depth-1)
            for my_rect in rects:   
                a = int((round(my_rect[0]/scale_back)))
                b = int((round(my_rect[1]/scale_back)))
                c = int((round(my_rect[2]/scale_back)))
                d = int((round(my_rect[3]/scale_back)))
                print c-a , d-b
                cv2.rectangle(my_im,(a,b),(c,d),(255,0,0),1)
                plt.imshow(my_im)

    plt.show()