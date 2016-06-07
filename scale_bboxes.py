import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import cv2
import numpy as np
import pickle
import image_pyramid as ip
import display_bboxes as db
import nonmax_supress as ns
import matplotlib.pyplot as plt
import math

def scale_bboxes(bboxes_dict, scale):
    bboxes = []
    for depth,rects in bboxes_dict.iteritems():
        if depth == 1:
            for rect in rects:
                bboxes.append(rect)

        else:
            scale_back = math.pow(scale, depth-1)
            for my_rect in rects:   
                a = int((round(my_rect[0]*scale_back)))
                b = int((round(my_rect[1]*scale_back)))
                c = int((round(my_rect[2]*scale_back)))
                d = int((round(my_rect[3]*scale_back)))
                new_box = [a,b,c,d]
                bboxes.append(new_box)
    return bboxes