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
                a = int((round(my_rect[0]/scale_back)))
                b = int((round(my_rect[1]/scale_back)))
                c = int((round(my_rect[2]/scale_back)))
                d = int((round(my_rect[3]/scale_back)))
                new_box = [a,b,c,d]
                bboxes.append(new_box)
    return bboxes


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
                    #print score_calc[0][0]
                    if(score_calc[0][0] >= 1):
                        print score_calc[0][0],curr_depth
                        scores.append(score_calc[0][0])
                        box = [w, h, w+window_size[1], h+window_size[0]]
                        if curr_depth in bboxes:
                            bboxes[curr_depth].append(box)
                            
                        else:
                            bboxes[curr_depth] = [box]
                            

    scaled_bboxes = scale_bboxes(bboxes, scale)
    return scaled_bboxes,scores


def run_detector(test_image_path):

    svm_model = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/trained_svm_model.p", "r"))
    window_size = [70, 35]
    scale = 0.95
    block_size = 2
    cell_size = 16
    min_height = 70
    min_width = 35

    return detector(test_image_path, svm_model, scale, min_height, min_width, block_size, cell_size, window_size)



#test_image_path = "/Users/azarf/Desktop/test_resize.pnm"
test_image_path = "/Users/azarf/Desktop/test_6.jpg"
#test_image_path = "/Users/azarf/Documents/Courses/Spring2016/CS231A/project/INRIAPerson/Train/pos/crop_000603.png"
#test_image_path = "/Users/azarf/Documents/Courses/Spring2016/CS231A/project/64x80/NICTA_Pedestrian_Positive_Valid_Set_A/00000002/item_00002005.pnm"
bboxes, scores = run_detector(test_image_path)
scale = 0.8
db.draw_bboxes(bboxes, test_image_path, scale)
print len(bboxes), len(scores)
print "#### start finding max supress boxes .... ###"
bboxes = ns.nonmax_supress(bboxes, scores)

scale = 0.8
db.draw_bboxes(bboxes, test_image_path, scale)


            
