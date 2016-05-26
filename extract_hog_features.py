
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
				image = image[17:145, 16:80]
				#cv2.imshow('test',image)
				#cv2.waitKey(0)
				my_feature, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=True)
				features.append(my_feature)
	return features

def extract_pos_hog_features2(path, num_samples):
	features = []
	cnt = 0
	for dirpath, dirnames, filenames in walk(path):
		for my_file in filenames:
			if cnt < num_samples:
				cnt = cnt + 1
				im = cv2.imread(path + my_file)
				image = color.rgb2gray(im)
				my_feature, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=True)
				features.append(my_feature)
	return features


def extract_neg_hog_features(path, num_samples):

	features = []
	cnt = 0
	for dirpath, dirnames, filenames in walk(path):
		for my_file in filenames:
			if cnt < num_samples:
				cnt = cnt + 1
				im = cv2.imread(path + my_file)
				image = color.rgb2gray(im)
				image = image[17:145, 16:80]
				#cv2.imshow('test',image)
				#cv2.waitKey(0)
				my_feature, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=True)
				features.append(my_feature)
	return features


def neg_hog_rand(path, num_samples, window_size, num_window_per_image):
	rows = window_size[0]
	cols = window_size[1]
	features = []
	cnt = 0
	for dirpath, dirnames, filenames in walk(path):
		for my_file in filenames:
			
			if cnt < num_samples:
				print cnt,my_file
				cnt = cnt + 1
				im = cv2.imread(path + my_file)
				image = color.rgb2gray(im)
				image_rows = image.shape[0]
				image_cols = image.shape[1]
				
				for i in range(0,num_window_per_image):
					x_min = random.randrange(0,image_rows - rows)
					y_min = random.randrange(0,image_cols - cols)

					x_max = x_min + rows
					y_max = y_min + cols
					
					image_hog = image[x_min:x_max , y_min:y_max]
					#cv2.imshow('test',image_hog)
					#cv2.waitKey(0)
					my_feature, _ = hog(image_hog, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=True)
					features.append(my_feature)
	return features


image_path_pos = "C:/Users/iarev1et/Desktop/pedestrian_final/pos/"
pos_features = extract_pos_hog_features(image_path_pos, 4400) #4400

image_path_pos2 = "C:/Users/iarev1et/Desktop/pedestrian_final/pos2/"
pos_features2 = extract_pos_hog_features2(image_path_pos2, 2000) #2000

pos_features = pos_features + pos_features2
print "finished positive"

#extract hog features for negative samples
#image_path_neg = "C:/Users/iarev1et/Desktop/pedestrian_final/neg/"
#neg_features = extract_neg_hog_features(image_path_neg, 100) #997
#print neg_features
#print 'finished negative'

image_path_neg2 = "C:/Users/iarev1et/Desktop/pedestrian_final/neg2/"
neg_features_rand = neg_hog_rand(image_path_neg2, 2100, [128,64], 10)

#print "finished neg rand"
# concatinate positive and negative hog featuress
features = pos_features + neg_features_rand
pickle.dump(features, open("C:/Users/iarev1et/Desktop/Python/ped_detector/ped_features.p", 'w'))

# assing labels to positive and negative samples
pos_labels = [1] * len(pos_features)
#neg_labels = [-1] * len(neg_features + neg_features_rand)
neg_labels = [-1] * len(neg_features_rand)

labels = pos_labels + neg_labels
pickle.dump(labels, open("C:/Users/iarev1et/Desktop/Python/ped_detector/peds_feature_to_label.p", 'w'))

print "done hog extraction"
