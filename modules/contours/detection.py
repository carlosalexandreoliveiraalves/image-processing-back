import cv2

def detect_edges_canny(image, params={}):
    threshold1 = params.get('threshold1', 100)
    threshold2 = params.get('threshold2', 200)
    return cv2.Canny(image, threshold1, threshold2)