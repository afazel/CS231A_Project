from sklearn import svm
from skimage import data, color, exposure
import pickle
import cv2
from skimage.feature import hog
from os import walk


# load traine model
svm_model = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/trained_svm_model.p", "r"))

path = "/Users/azarf/Desktop/validation/"
test_features = []
for dirpath, dirnames, filenames in walk(path):
	for my_file in filenames:
		print my_file
		im = cv2.imread(path + my_file)
		image = color.rgb2gray(im)
		my_feature, _ = hog(image, orientations = 9, pixels_per_cell = (8, 8),cells_per_block = (4, 4), visualise = True)
		test_features.append(my_feature)
# load test features
#test_features = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/ped_test_features.p","r"))

# predict labels
svm_model.predict(test_features)

# compute accuracy
print svm_model.score(test_features, [1]*len(test_features), sample_weight=None)