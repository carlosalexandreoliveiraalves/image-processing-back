import cv2

def apply_gaussian_blur(image, params={}):
    ksize = params.get('ksize', 5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)