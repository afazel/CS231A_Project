import cv2
import is_pedestrian as ped
import math
from skimage import color
import pickle
import nonmax_supress as ns

svm_model = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_Project/trained_svm_model.p", "r"))
weight = svm_model.coef_ 
bias = svm_model.intercept_

cap = cv2.VideoCapture("/Users/azarf/Desktop/testvids/terrace.mp4")
fgbg = cv2.BackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    print "################ NEW FRAME #################"
    bbox = []
    scores = []
    rect_frame = frame
    fgmask = fgbg.apply(frame)

    cv2.imshow('Background Subtraction',fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cnt = 0
    (contours, _) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (xmin, ymin, w, h) = cv2.boundingRect(c)
        if w > h:
            continue 
        
        window_size = [64, 128]
        xmax = xmin + w
        ymax = ymin + h
        if w < window_size[0]:
            padding_width = window_size[0] - w
            xmin_p = xmin-math.ceil(padding_width / 2.0) 
            xmax_p = xmax + math.ceil(padding_width / 2.0)
            if h < window_size[1]:
                 padding_height = window_size[1] - h
                 ymin_p = ymin - math.ceil(padding_height / 2.0)
                 ymax_p = ymax + math.ceil(padding_height / 2.0)
         
        if xmin_p < 0 or xmax_p > frame.shape[1] or ymin_p < 0 or ymax_p > frame.shape[0]:
            continue

        extracted_contour = frame[ymin_p:ymax_p , xmin_p:xmax_p]
        extracted_contour = color.rgb2gray(extracted_contour)  
        
        scale = 1.2
        score = ped.detector(extracted_contour,weight,bias,scale)
        if score:
            scores.append(score)
            bbox.append([int(xmin_p), int(ymin_p), int(xmax_p), int(ymax_p)])
            
    if scores:
        bbox = ns.nonmax_supress(bbox, scores)

    for box in bbox:
        cv2.rectangle(rect_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow("Video",rect_frame)
    cv2.waitKey(1)
        

cap.release()
cv2.destroyAllWindows()