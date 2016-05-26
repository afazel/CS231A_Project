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


def detector(test_image_path, svm_model, scale, min_height, min_width, block_size, cell_size, window_size,orient, thresh,flag):
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
        hard_minig_cnt = 0
        for h in xrange(0,H,total_block_size / 2):
            for w in xrange(0,W,total_block_size / 2):
                if ((window_size[1] + w <= W) and (window_size[0]+h) <= H):
                    fd, hog_im = hog(im[h:(window_size[0]+h), w:(window_size[1]+w)], orientations=orient, pixels_per_cell=(cell_size, cell_size),
                    cells_per_block=(block_size, block_size), visualise=True)
                    #cv2.imshow('test',im[h:(window_size[0]+h), w:(window_size[1]+w)])
                    #cv2.waitKey(0)
                    score_calc =  np.dot(np.reshape(fd, (1, dim_size_feat)) , np.transpose(weight)) + bias
                    #print score_calc[0][0], curr_depth
                    if(score_calc[0][0] >= thresh):
                        print "score and depth: ", score_calc[0][0],curr_depth
                        if flag == 1:
                            scores.append(score_calc[0][0])
                            box = [w, h, w+window_size[1], h+window_size[0]]
                            if curr_depth in bboxes:
                                bboxes[curr_depth].append(box)
                            
                            else:
                                bboxes[curr_depth] = [box]
                        else: # do nagative hard mining
                            new_image = im[h:(window_size[0]+h), w:(window_size[1]+w)]
                            plt.imshow(new_image) #Needs to be in row,col order
                            plt.savefig("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/negative_hard_mining/nhm_"+hard_minig_cnt+"jpg")
                            hard_minig_cnt += 1
    if flag == 1:
        scaled_bboxes = scale_bboxes(bboxes, scale)
        return scaled_bboxes,scores


def run_detector(test_image_path,scale, flag):

    svm_model = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_Project/trained_svm_model.p", "r"))
    window_size = [128, 64]
    block_size = 2
    cell_size = 8
    min_height = 128
    min_width = 64
    orient = 9

    return detector(test_image_path, svm_model, scale, min_height, min_width, block_size, cell_size, window_size,orient, flag)

########################################################################################
scale = 1.2
#test_image_path = "C:/Users/iarev1et/Desktop/pedestrians.jpg"
#test_image_path = "C:/Users/iarev1et/Desktop/inria2.png"
#test_image_path = "C:/Users/iarev1et/Desktop/test2.jpg"
#test_image_path = "C:/Users/iarev1et/Desktop/ped2.jpg"
#test_image_path = "C:/Users/iarev1et/Desktop/person_323.png"
#test_image_path = "C:/Users/iarev1et/Desktop/ped1.jpg"
#test_image_path = "C:/Users/iarev1et/Desktop/test8.jpg"
#test_image_path = "C:/Users/iarev1et/Desktop/person_217.png"
flag = 0
if flag == 1:
    bboxes, scores = run_detector(test_image_path,scale,flag)

    db.draw_bboxes(bboxes, test_image_path, scale)
    print len(bboxes), len(scores)
    print "#### start finding max supress boxes .... ###"
    bboxes = ns.nonmax_supress(bboxes, scores)
    db.draw_bboxes(bboxes, test_image_path, scale)
else:
    test_image_path = "/Users/azarf/Desktop/negtest3.jpg"
    run_detector(test_image_path,scale,flag)


            
