import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import cv2
import numpy as np
import pickle
import imagePyramid as ip
import display_bboxes as db
import matplotlib.pyplot as plt
import math

def detector(test_image_path, svm_model, scale, min_height, min_width, block_size, cell_size, window_size):
    weight = svm_model.coef_ 
    bias = svm_model.intercept_

    my_im = cv2.imread(test_image_path)
    my_im = color.rgb2gray(my_im)

    total_block_size = block_size * cell_size; 

    scores = []
    bboxes = {}
    curr_depth = 0


    for im in ip.createImagePyramid(my_im, scale, min_height, min_width):
        curr_depth +=1
        H = im.shape[0]
        W = im.shape[1]
    
        dim_size_feat = weight.shape[1];

        for h in xrange(0,H,total_block_size / 2):
            for w in xrange(0,W,total_block_size / 2):
                if ((window_size[1] + w <= W) and (window_size[0]+h) <= H):
                    fd, hog_im = hog(im[h:(window_size[0]+h), w:(window_size[1]+w)], orientations=8, pixels_per_cell=(cell_size, cell_size),
                    cells_per_block=(block_size, block_size), visualise=True)
                
                    score_calc =  np.dot(np.reshape(fd, (1, dim_size_feat)) , np.transpose(weight)) + bias
                    if(score_calc[0][0] >= 0.5):
                        scores.append(score_calc[0][0])
                        box = [w, h, w+window_size[1], h+window_size[0]]
                        if curr_depth in bboxes:
                            bboxes[curr_depth].append(box)
                        else:
                            bboxes[curr_depth] = [box]


    return bboxes


def run_detector(test_image_path):

    svm_model = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/trained_svm_model.p", "r"))
    window_size = [80, 64]
    scale = 0.8
    block_size = 2
    cell_size = 16
    min_height = 64
    min_width = 80

    return detector(test_image_path, svm_model, scale, min_height, min_width, block_size, cell_size, window_size)



test_image_path = "/Users/azarf/Desktop/ped_test_2.jpeg"
bboxes = run_detector(test_image_path)
scale = 0.8
db.draw_bboxes(bboxes, test_image_path, scale)


            
