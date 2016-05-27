import cv2
cap = cv2.VideoCapture("/Users/azarf/Desktop/passageway_test.mp4")
#passageway_test.mp4
#Recording-Session-840674.mp4
ret, frame = cap.retrieve()
#print frame[0]

#Recording-Session-840674.mp4

# while True:
#     if cap.grab():

#         #flag, frame = cap.retrieve()
#         flag, frame = cap.read()
#         print "here",frame
#         if not flag:
#             continue
#         else:
#             cv2.imshow('video', frame)
#     if cv2.waitKey(10) == 27:
#         break

fgbg = cv2.BackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    #print frame

    fgmask = fgbg.apply(frame)
    #print fgmask

    cv2.imshow('frame',fgmask)
    cv2.imshow('img',frame)
    # if cv2.waitKey(10) == 27:
    #      break
    k = cv2.waitKey(30) & 0xff
    if k == 5:
    	break

    (cnts, _) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
		# if the contour is too small, ignore it
		#print cv2.contourArea(c)
		if cv2.contourArea(c) < 100:
			continue
 
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		print x+w,y+h
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.imshow("detect",frame)
		cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()