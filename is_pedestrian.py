from skimage.feature import hog
from skimage import color
import cv2
import numpy as np
import pickle
import image_pyramid as ip


def detector(my_im, weight,bias, scale, min_height, min_width, block_size, cell_size, window_size,orient, thresh):
    #weight = svm_model.coef_ 
    #bias = svm_model.intercept_
    print "I am here"
    #my_im = cv2.imread(test_image_path)
    #my_im = color.rgb2gray(my_im)

    total_block_size = block_size * cell_size; 
    curr_depth = 0

    for im in ip.createImagePyramid(my_im, scale, min_height, min_width):
        curr_depth +=1
        H = im.shape[0]
        W = im.shape[1]
        dim_size_feat = weight.shape[1];
        for h in xrange(0,H,total_block_size / 2):
            for w in xrange(0,W,total_block_size / 2):
                if ((window_size[1] + w <= W) and (window_size[0]+h) <= H):
                    fd, _ = hog(im[h:(window_size[0]+h), w:(window_size[1]+w)], orientations=orient, pixels_per_cell=(cell_size, cell_size),
                    cells_per_block=(block_size, block_size), visualise=True)
                
                    score_calc =  np.dot(np.reshape(fd, (1, dim_size_feat)) , np.transpose(weight)) + bias
                    print score_calc[0][0]
                    if(score_calc[0][0] >= thresh):
                        return True
    return False


def run_detector(image,weight, bias, scale):
    #svm_model = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_Project/trained_svm_model.p", "r"))
    window_size = [128, 64]
    block_size = 2
    cell_size = 8
    min_height = 128
    min_width = 64
    orient = 9
    thresh = 0
    #return detector(image, svm_model, scale, min_height, min_width, block_size, cell_size, window_size,orient,thresh)
    return detector(image, weight, bias, scale, min_height, min_width, block_size, cell_size, window_size,orient,thresh)
            