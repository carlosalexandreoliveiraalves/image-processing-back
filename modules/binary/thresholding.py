import cv2


def apply_threshold(image, params={}):
    thresh_val = params.get('thresh_val', 127)
    max_val = params.get('max_val', 255)
    _, binary_img = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_BINARY)
    return binary_img


def apply_threshold_inv(image, params={}):
    thresh_val = params.get('thresh_val', 127)
    max_val = params.get('max_val', 255)
    _, binary_img = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_BINARY_INV)
    return binary_img


def apply_threshold_trunc(image, params={}):
    thresh_val = params.get('thresh_val', 127)
    max_val = params.get('max_val', 255)
    _, binary_img = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_TRUNC)
    return binary_img


def apply_threshold_tozero(image, params={}):
    thresh_val = params.get('thresh_val', 127)
    max_val = params.get('max_val', 255)
    _, binary_img = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_TOZERO)
    return binary_img


def apply_threshold_tozero_inv(image, params={}):
    thresh_val = params.get('thresh_val', 127)
    max_val = params.get('max_val', 255)
    _, binary_img = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_TOZERO_INV)
    return binary_img