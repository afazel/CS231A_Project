
from skimage.feature import hog
from skimage import data, color, exposure
import cv2
import os
from os import walk
import pickle
import random

def extract_pos_hog_features(path, num_samples):

	features = []
	cnt = 0
	for dirpath, dirnames, filenames in walk(path):
		for my_file in filenames:
			if cnt < num_samples:
				cnt = cnt + 1
				im = cv2.imread(path + my_file)
				image = color.rgb2gray(im)
				my_feature, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(2, 2), visualise=True)
				features.append(my_feature)
	return features


def extract_neg_hog_features(path, num_samples, window_size, num_window_per_image):
	rows = window_size[0]
	cols = window_size[1]
	features = []
	cnt = 0
	for dirpath, dirnames, filenames in walk(path):
		for my_file in filenames:
			if cnt < num_samples:
				cnt = cnt + 1
				im = cv2.imread(path + my_file)
				print path + my_file
				image = color.rgb2gray(im)
				image_rows = image.shape[0]
				image_cols = image.shape[1]
				
				for i in range(0,num_window_per_image):
					x_min = random.randrange(0,image_rows - rows)
					y_min = random.randrange(0,image_cols - cols)

					x_max = x_min + rows
					y_max = y_min + cols

					image_hog = image[x_min:x_max , y_min:y_max]

					my_feature, _ = hog(image_hog, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(2, 2), visualise=True)

					print my_feature.shape
					features.append(my_feature)
	return features



image_path_pos = "/Users/azarf/Documents/Courses/Spring2016/CS231A/project/64x80/NICTA_Pedestrian_Positive_Train_Set_A/00000000/"
pos_features = extract_pos_hog_features(image_path_pos, 1000)

#extract hog features for negative samples
image_path_neg = "/Users/azarf/Documents/Courses/Spring2016/CS231A/project/INRIAPerson/Train/neg/"
neg_features = extract_neg_hog_features(image_path_neg, 200, [64,80], 5)


# concatinate positive and negative hog features
features = pos_features + neg_features
pickle.dump(features, open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/ped_features.p", 'w'))

# assing labels to positive and negative samples
pos_labels = [1] * len(pos_features)
neg_labels = [-1] * len(neg_features)

labels = pos_labels + neg_labels
pickle.dump(labels, open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/peds_feature_to_label.p", 'w'))


#extract hog features for testing
image_path_pos_test = "/Users/azarf/Documents/Courses/Spring2016/CS231A/project/64x80/NICTA_Pedestrian_Positive_Valid_Set_A/00000000/"
pos_test_features = extract_pos_hog_features(image_path_pos, 1000)
pickle.dump(pos_test_features, open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/ped_test_features.p", 'w'))

