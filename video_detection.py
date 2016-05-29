import cv2
import is_pedestrian as ped
import math
from skimage import color
import pickle

svm_model = pickle.load(open("/Users/azarf/Documents/Courses/Spring2016/CS231A/project/CS231A_Project/trained_svm_model.p", "r"))
weight = svm_model.coef_ 
bias = svm_model.intercept_

cap = cv2.VideoCapture("/Users/azarf/Desktop/testvids/basketball.mp4")
#passageway_test.mp4
#Recording-Session-840674.mp4
#ped_vid_good_azar.mp4
#terrace.mp4"
#WalkByShop1front.mp4
#ret, frame = cap.retrieve()
fgbg = cv2.BackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    print "################ NEW FRAME #################"
    #print "frame len",len(frame)
    bbox = []
    rect_frame = frame
    #print frame.shape
    fgmask = fgbg.apply(frame)
    #cv2.imshow('frame',fgmask)
    #cv2.imshow('img',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
    cnt = 0
    (contours, _) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print "contour length",len(contours)
    for c in contours:
        #print "contour size:", c.shape
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 10:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (xmin, ymin, w, h) = cv2.boundingRect(c)
        
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

        print "original contour:", xmin, ymin , h, w
        extracted_contour = frame[ymin_p:ymax_p , xmin_p:xmax_p]
        print "resized contour:" , extracted_contour.shape
        #cv2.imshow("current image", extracted_contour)
        #cv2.waitKey(25)
        #print extracted_contour
        #extracted_contour = cv2.imread(extracted_contour)
        extracted_contour = color.rgb2gray(extracted_contour)  
        
        scale = 1.2
        my_ped = ped.run_detector(extracted_contour,weight,bias,scale)
        #print ped
        if my_ped:
            cnt +=1
            #bbox.append([int(xmin), int(ymin), int(xmin + w), int(ymin + h)])
            bbox.append([int(xmin_p), int(ymin_p), int(xmax_p), int(ymax_p)])
            #cv2.rectangle(rect_frame, (int(xmin), int(ymin)), (int(xmin + w), int(ymin + h)), (0, 255, 0), 2)
            #print "find ped"
            #print xmin, ymin,xmin + w, ymin + h
    print "# boxes:",len(bbox)
    print "# peds", cnt
    print "# contours", len(contours)
    for box in bbox:
        cv2.rectangle(rect_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow("detect",rect_frame)
    cv2.waitKey(1)
        

cap.release()
cv2.destroyAllWindows()