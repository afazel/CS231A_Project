from sklearn import svm
from skimage import data, color, exposure
import pickle
import cv2
from skimage.feature import hog


def train_svm(features, labels, reg_param, kernel_type):
	clf = svm.SVC(C = reg_param, kernel = kernel_type)
	svm_model = clf.fit(features, labels) 
	print "fitting model done!"
    return svm_model


features = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/ped_features.p", "r"))
labels = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/peds_feature_to_label.p", "r"))

svm_model = train_svm(features, labels, 0.05, "linear")
pickle.dump(svm_model, open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_project/trained_svm_model.p", "w"))

