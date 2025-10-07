import cv2

def to_grayscale(image, params={}):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_hsv(image, params={}):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)