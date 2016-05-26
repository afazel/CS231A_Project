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


features = pickle.load(open("C:/Users/iarev1et/Desktop/Python/ped_detector/ped_features.p", "r"))
labels = pickle.load(open("C:/Users/iarev1et/Desktop/Python/ped_detector/peds_feature_to_label.p", "r"))
svm_model = train_svm(features, labels, 0.01,'linear')
pickle.dump(svm_model, open("C:/Users/iarev1et/Desktop/Python/ped_detector/trained_svm_model.p", "w"))


