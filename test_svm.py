from sklearn import svm
from skimage import data, color, exposure
import pickle
import cv2
from skimage.feature import hog


# load traine model
svm_model pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/trained_svm_model.p", "r"))

# load test features
test_features = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/ped_test_features.p","r"))

# predict labels
model.predict(test_features)

# compute accuracy
print model.score(test_features, [1]*len(test_features), sample_weight=None)