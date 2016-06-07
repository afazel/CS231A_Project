import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import cv2
import numpy as np
import pickle
import image_pyramid as ip
import matplotlib.pyplot as plt
import math


def draw_bboxes(bboxes, image_path, scale):

    my_im = cv2.imread(image_path)
    for box in bboxes:
        cv2.rectangle(my_im,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
    
    cv2.imshow('test',my_im)
    cv2.waitKey(1)

