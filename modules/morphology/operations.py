import cv2
import numpy as np


def apply_erosion(image, params={}):
    kernel_size = params.get('kernel_size', 5)
    iterations = params.get('iterations', 1)  # <-- MUDANÇA AQUI: Lê as iterações dos parâmetros
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)  # <-- MUDANÇA AQUI: Usa a variável


def apply_dilation(image, params={}):
    kernel_size = params.get('kernel_size', 5)
    iterations = params.get('iterations', 1)  # <-- MUDANÇA AQUI: Lê as iterações dos parâmetros
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)  # <-- MUDANÇA AQUI: Usa a variável


def apply_open(image, params={}):
    kernel_size = params.get('kernel_size', 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def apply_close(image, params={}):
    kernel_size = params.get('kernel_size', 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def apply_grad(image, params={}):
    kernel_size = params.get('kernel_size', 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def apply_tophat(image, params={}):
    kernel_size = params.get('kernel_size', 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def apply_blackhat(image, params={}):
    kernel_size = params.get('kernel_size', 5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
