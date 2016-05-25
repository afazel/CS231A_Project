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
        cv2.rectangle(my_im,(box[0],box[1]),(box[2],box[3]),(255,0,0),1)
        plt.imshow(my_im)

    plt.show()