from sklearn import svm
from skimage import data, color, exposure
import pickle
import cv2
from skimage.feature import hog
from os import walk


# load traine model
rf_model = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/models/trained_rf_model_4_16_1000.p", "r"))

path = "/Users/azarf/Desktop/validation/pos/"
test_features = []
for dirpath, dirnames, filenames in walk(path):
	for my_file in filenames:
		print my_file
		im = cv2.imread(path + my_file)
		image = color.rgb2gray(im)
		my_feature, _ = hog(image, orientations = 9, pixels_per_cell = (16, 16),cells_per_block = (4, 4), visualise = True)
		test_features.append(my_feature)

# predict labels
rf_model.predict(test_features)

# compute accuracy
print rf_model.score(test_features, [1]*len(test_features), sample_weight=None)