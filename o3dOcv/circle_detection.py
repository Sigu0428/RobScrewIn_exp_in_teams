from math import pow
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

def fitness(points, edge_img):
    x1, y1 = points[0, :]
    x2, y2 = points[1, :]
    x3, y3 = points[2, :]

    A = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
    B = (x1**2 + y1**2)*(y3-y2) + (x2**2 + y2**2)*(y1-y3) + (x3**2 + y3**2)*(y2-y1)
    C = (x1**2 + y1**2)*(x2-x3) + (x2**2 + y2**2)*(x3-x1) + (x3**2 + y3**2)*(x1-x2)
    D = (x1**2 + y1**2)*(x3*y2 - x2*y3) + (x2**2 + y2**2)*(x1*y3 - x3*y1) + (x3**2 + y3**2)*(x2*y1 - x1*y2)
    
    xc = np.round(-B/(2*A)).astype(int)
    yc = np.round(-C/(2*A)).astype(int)
    R = np.round(np.sqrt((B**2 + C**2 - 4*A*D)/(4*(A**2)))).astype(int)
    
    coords = ski.draw.circle_perimeter(xc, yc, R, shape=edge_img.shape)
    return np.sum(edge_img[coords])


image = ski.io.imread('circle_test.jpg')
image = ski.color.rgb2gray(image)
edges_img = canny(image, sigma=3)
labeled_image, count = ski.measure.label(edges_img, return_num=True)
rgb_labels = ski.color.label2rgb(labeled_image, bg_label=0)
#plotImage("rgb_labels")

# POTENTIAL BUG ALLEERT, EACH RIBON MIGHT BE UNSORTED WHICH COULD CAUSE PROBLEMS WHEN FINDING THE MIDDLE OF A RIBBON
labeled_image = np.array(labeled_image)
edges = np.zeros((0, 2))
for label in range(2, count+1): # By starting at 2 we miss the background label 0 and edge caused by zero padding with label 1
    x, y = np.where(labeled_image == label)
    edges = np.block([[edges], [x[:, None], y[:, None]]])

indices = np.random.randint(0, edges.shape[0], 3)
fit = fitness(edges[indices, :], edges_img)
print(fit)