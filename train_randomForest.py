from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pickle


features = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/features/ped_features_4_16.p", "r"))
labels = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_Project/peds_feature_to_label.p", "r"))

rf = RandomForestClassifier(n_estimators=1000)
rf_model = rf.fit(features,labels)

pickle.dump(rf_model, open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_Project/models/trained_rf_model_4_16_1000.p", "w"))