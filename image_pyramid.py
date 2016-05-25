import imutils
import cv2
import matplotlib.pyplot as plt

def createImagePyramid(im, scale, minHeight, minWidth):
	yield im
	while (True):
		reducedWidth = int(round(im.shape[1] * scale))
		reducedHeight = int(round(im.shape[0] * scale))
		im = imutils.resize(im, width=reducedWidth, height=reducedHeight)
		if ((im.shape[0] >= minHeight) and (im.shape[1] >= minWidth)):
			yield im

		else:
			break
    
