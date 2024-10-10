from math import pow
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np
from numpy.linalg import *
from scipy.ndimage import convolve
import cv2 as cv
from objective_function_interface import *
from harmony_search import *
import random
from multiprocessing import cpu_count
from bisect import bisect_left

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

class ObjectiveFunc(ObjectiveFunctionInterface):
    def __init__(self, image):
        image = ski.color.rgb2gray(image)
        self._edge_img = canny(image, mode='reflect', sigma=3)
        self._sorted_edges = get_edges(self._edge_img)
        self._upper_bounds = [0, self._sorted_edges.shape[0]-1]
        self._lower_bounds = [0, self._sorted_edges.shape[0]-1]
        self._variable = [True, True]
        self._discrete_values = [[x for x in range(0, self._sorted_edges.shape[0])], [x for x in range(0, self._sorted_edges.shape[0])]]

        # define all input parameters
        self._maximize = True  # do we maximize or minimize?
        self._max_imp = 5000  # maximum number of improvisations
        self._hms = 100  # harmony memory size
        self._hmcr = 0.75  # harmony memory considering rate
        self._par = 0.5  # pitch adjusting rate
        self._mpap = 0.25  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
        self._mpai = 2  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only
        self._random_seed = 8675309  # optional random seed for reproducible results

    def solution_vec_to_points(self, x):
        idx1 = x[0]
        idx3 = x[1]
        idx2 = (int)((idx1 + idx3) / 2)

        p1 = self._sorted_edges[idx1, 0:2]
        p2 = self._sorted_edges[idx2, 0:2]
        p3 = self._sorted_edges[idx3, 0:2]
        
        return (p1, p2, p3)

    def get_fitness(self, vector):
        p1, p2, p3 = self.solution_vec_to_points(vector)

        fit = ObjectiveFunc.fit_circle(p1, p2, p3)
        if fit is None:
            return 0
        else:
            xc, yc, R = fit
        coords = ski.draw.circle_perimeter(xc, yc, R, shape=self._edge_img.shape)
        obj_func.visualize_circle(p1, p2, p3, "fitness" + str(np.sum(self._edge_img[coords])))
        return np.sum(self._edge_img[coords])

    def get_value(self, i, j=None):
        if self.is_discrete(i):
            if j:
                return self._discrete_values[i][j]
            return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

    @staticmethod
    def binary_search(a, x):
        """
            Code courtesy Python bisect module: http://docs.python.org/2/library/bisect.html#searching-sorted-lists
        """
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        raise ValueError
    
    def get_index(self, i, v):
        """
            Because self.discrete_values is in sorted order, we can use binary search.
        """
        return ObjectiveFunc.binary_search(self._discrete_values[i], v)

    def get_lower_bound(self, i):
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        return self._upper_bounds[i]

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        # all variables are continuous
        return True
    
    def get_num_discrete_values(self, i):
        if self.is_discrete(i):
            return len(self._discrete_values[i])
        return float('+inf')

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return True
    
    def get_random_seed(self):
        return self._random_seed

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize
    
    @staticmethod
    def fit_circle(p1, p2, p3):
        A = p1[0]*(p2[1]-p3[1]) - p1[1]*(p2[0]-p3[0]) + p2[0]*p3[1] - p3[0]*p2[1]
        B = (p1[0]**2 + p1[1]**2)*(p3[1]-p2[1]) + (p2[0]**2 + p2[1]**2)*(p1[1]-p3[1]) + (p3[0]**2 + p3[1]**2)*(p2[1]-p1[1])
        C = (p1[0]**2 + p1[1]**2)*(p2[0]-p3[0]) + (p2[0]**2 + p2[1]**2)*(p3[0]-p1[0]) + (p3[0]**2 + p3[1]**2)*(p1[0]-p2[0])
        D = (p1[0]**2 + p1[1]**2)*(p3[0]*p2[1] - p2[0]*p3[1]) + (p2[0]**2 + p2[1]**2)*(p1[0]*p3[1] - p3[0]*p1[1]) + (p3[0]**2 + p3[1]**2)*(p2[0]*p1[1] - p1[0]*p2[1])
        if A == 0.0:
            return None
        xc = np.round(-B/(2*A)).astype(int)
        yc = np.round(-C/(2*A)).astype(int)
        R = np.round(np.sqrt((B**2 + C**2 - 4*A*D)/(4*(A**2)))).astype(int)
        return (xc, yc, R)
    
    def visualize_circle(self, p1, p2, p3, title=None):
        img = ski.util.img_as_float(self._edge_img)
        img = ski.color.gray2rgb(img)
        xc, yc, R = ObjectiveFunc.fit_circle(p1, p2, p3)
        img[ski.draw.circle_perimeter(xc, yc, R, shape=self._edge_img.shape)] = [255, 0, 0]
        img[ski.draw.disk((p1[0], p1[1]), 5, shape=img.shape)] = [0, 255, 0]
        img[ski.draw.disk((p2[0], p2[1]), 5, shape=img.shape)] = [0, 255, 0]
        img[ski.draw.disk((p3[0], p3[1]), 5, shape=img.shape)] = [0, 255, 0]
        if title is not None:
            plotImage(img, title)
        else:
            plotImage(img, "circle")
    
def get_edges(edges_img):
    labeled_image, count = ski.measure.label(edges_img, return_num=True)

    # POTENTIAL BUG ALLEERT, EACH RIBON MIGHT BE UNSORTED WHICH COULD CAUSE PROBLEMS WHEN FINDING THE MIDDLE OF A RIBBON
    labeled_image = np.array(labeled_image)
    edges = np.zeros((0, 4))
    for label in range(1, count+1): # By starting at 2 we miss the background with label 0
        x, y = np.where(labeled_image == label)
        ribb_min_lim = edges.shape[0]*np.ones((x.size, 1))
        ribb_max_lim = (edges.shape[0]+x.size)*np.ones((x.size, 1))
        edges = np.block([[edges], [x[:, None], y[:, None], ribb_min_lim, ribb_max_lim]])
    return edges

def plotImage(image, title):
    if(len(image.shape) == 2):
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

image = ski.io.imread('./images/image18.jpg')
obj_func = ObjectiveFunc(image)
num_processes = cpu_count()
num_iterations = num_processes
results = harmony_search(obj_func, num_processes, num_iterations)
print('Elapsed time: {}\nBest harmony: {}\nBest fitness: {}\nHarmony memories:'.format(results.elapsed_time, results.best_harmony, results.best_fitness))

p1, p2, p3 = obj_func.solution_vec_to_points(results.best_harmony)
obj_func.visualize_circle(p1, p2, p3)