import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import cv2
import numpy as np


im = cv2.imread("C:/Users/iarev1et/Desktop/00089.jpg")
image_gray = color.rgb2gray(im)

#cv2.imshow('image',image_gray)
#fd, hog_image = hog(image_gray , orientations=8, pixels_per_cell=(16, 16),
                    #cells_per_block=(2, 2), visualise=True)

block_size = 2
cell_size = 8
total_block_size = block_size * cell_size; 

H = im.shape[0]
W = im.shape[1]
window_size = [64, 80]
dim_size_feat = weight.shape[0];
scores = []
bboxes = []

for h in range(0,total_block_size / 2):
    for w in range(0,total_block_size / 2):
        if ((window_size[1] + w <= W) and (window_size[0]+h) <= H):
            fd, hog_im = hog(im[h:(window_size[0]+h), w:(window_size[1]+w )], orientations=8, pixels_per_cell=(cell_size, cell_size),
                    cells_per_block=(block_size, block_size), visualise=True)
            score_calc =  np.reshape(fd, (1, dim_size_feat)) * weight + bias;
            if(score_calc >= 1):
                scores = scores.append(score_calc)
                box = [w, h, w+window_size[1],h+window_size[0]]
                bboxes = bboxes.append(box)