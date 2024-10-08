import math
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np
from numpy.linalg import *
from scipy.ndimage import convolve
import cv2 as cv

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

def plotImage(name):
    image = globals()[name]
    if(len(image.shape) == 2):
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(name)
    plt.axis('off')
    plt.show()

def objectie(x):
    pass

def fitness(pi, pj, pk, edge_map):
    xi, yi = pi
    xj, yj = pj
    xk, yk = pk
    Xji = xj - xi
    Xki = xk - xi
    Yji = yj - yi
    Yki = yk - yi
    A0 = xj**2 + yj**2 - xi**2 - yi**2
    A1 = xk**2 + yk**2 - xi**2 - yi**2
    A = np.block([[A0, 2*Yji], [A1, 2*Yki]])
    B = np.block([[2*Xji, A0], [2*Xki, A1]])
    D = 4*(Xji*Yki - Xki*Yji)

    x0 = det(A)/D
    y0 = det(B)/D
    r = np.sqrt((x0-xi)**2 + (y0-yi)**2)
    print(locals())
    coords = ski.draw.circle_perimeter(x0, y0, r)
    return np.sum(edge_map[coords])


image = ski.io.imread('circle_test.jpg')
image = ski.color.rgb2gray(image)
edges = canny(image, sigma=3)
labeled_image, count = ski.measure.label(edges, return_num=True)
rgb_labels = ski.color.label2rgb(labeled_image, bg_label=0)
#plotImage("rgb_labels")

# POTENTIAL BUG ALLEERT, EACH RIBON MIGHT BE UNSORTED WHICH COULD CAUSE PROBLEMS WHEN FINDING THE MIDDLE OF A RIBBON
labeled_image = np.array(labeled_image)
edges = np.zeros((0, 2))
for label in range(1, count+1):
    x, y = np.where(labeled_image == label)
    edges = np.block([[edges], [x[:, None], y[:, None]]])

fitness(edges[10, :], edges[20, :], edges[40, :], edges)