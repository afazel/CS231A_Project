import imutils
import cv2

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
    

im = cv2.imread("C:/Users/iarev1et/Desktop/00089.jpg")
scale = 0.50
minHeight = 64
minWidth = 80

for image in createImagePyramid(im, scale, minHeight, minWidth):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    print ("Current Height: ", image.shape[0])
    