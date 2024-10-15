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


# attributes is a dict of variable names and values.
# It will automatically generate a get_'name of variable' function, which the Objective function must implement to work anyway
# const_functions is the same but for function that need to be implemented but will always return the same value anyway
def add_attributes(attributes, const_functions):
    def class_decorator(cls):
        ''' Create getter and setter methods for each of the given attributes -answered Jan 9, 2020 at 7:33 by martineau on stack overflow'''
        for attr_name, inital_value in attributes.items(): # attributes that need a getter function (get_ + variable name)
            def getter(self, name=attr_name):
                return getattr(self, name)
            setattr(cls, attr_name, inital_value)
            setattr(cls, 'get_' + attr_name, getter)

        for func_name, result in const_functions.items(): # functions that always return the same value
            def func(self, name=func_name):
                return result
            setattr(cls, func_name, func)
        return cls
    return class_decorator

@add_attributes(
    attributes={'random_seed': random.randint(0, 10000), # optional random seed for reproducible results
                  'mpai': 2, # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only
                  'par': 0.5, # pitch adjusting rate
                  'max_imp': 5000,  # maximum number of improvisations
                  'hms': 100,  # harmony memory size
                  'hmcr': 0.75,  # harmony memory considering rate
                  'canny_low_threshold_quantile': 0.6, # between 0 and 1, e.g 0.6'th quantile
                  'canny_high_threshold_quantile': 0.9,
                  'num_parameters': 2 # size of the vector argument to the fitness function
                  },
    const_functions={'is_discrete': True,
                    'is_variable': True,
                    'use_random_seed': True,
                    'maximize': True,
                    })
class ObjectiveFunc(ObjectiveFunctionInterface):
    def __init__(self, image):
        self._image = ski.color.rgb2gray(image)
        self._edge_img = canny(self._image, mode='reflect', sigma=3, use_quantiles=True, low_threshold=self.canny_low_threshold_quantile, high_threshold=self.canny_high_threshold_quantile)
        self._sorted_edges = get_edges(self._edge_img)
        self._discrete_values = [[x for x in range(0, self._sorted_edges.shape[0])], [x for x in range(0, self._sorted_edges.shape[0])]]

    def solution_vec_to_points(self, x):
        idx1, idx3 = x
        idx2 = (int)((idx1 + idx3) / 2)
        p1, p2, p3 = self._sorted_edges[(idx1, idx2, idx3), 0:2]
        return (p1, p2, p3)

    def get_fitness(self, vector):
        p1, p2, p3 = self.solution_vec_to_points(vector)

        fit = ObjectiveFunc.fit_circle(p1, p2, p3)
        if fit is None:
            return 0
        else:
            xc, yc, R = fit
        coords = ski.draw.circle_perimeter(xc, yc, R, shape=self._edge_img.shape)
        inliers = np.sum(self._edge_img[coords])
        circumference = 2*np.pi*R
        fitness = inliers / np.float_power(circumference, 1/2)
        return fitness

    def get_value(self, i, j=None):
        if j:
            return self._discrete_values[i][j]
        return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]

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
        return 0

    def get_upper_bound(self, i):
        return self._sorted_edges.shape[0]-1
    
    def get_num_discrete_values(self, i):
        return len(self._discrete_values[i])
    
    @staticmethod
    def fit_circle(p1, p2, p3):
        A = p1[0]*(p2[1]-p3[1]) - p1[1]*(p2[0]-p3[0]) + p2[0]*p3[1] - p3[0]*p2[1]
        B = (p1[0]**2 + p1[1]**2)*(p3[1]-p2[1]) + (p2[0]**2 + p2[1]**2)*(p1[1]-p3[1]) + (p3[0]**2 + p3[1]**2)*(p2[1]-p1[1])
        C = (p1[0]**2 + p1[1]**2)*(p2[0]-p3[0]) + (p2[0]**2 + p2[1]**2)*(p3[0]-p1[0]) + (p3[0]**2 + p3[1]**2)*(p1[0]-p2[0])
        D = (p1[0]**2 + p1[1]**2)*(p3[0]*p2[1] - p2[0]*p3[1]) + (p2[0]**2 + p2[1]**2)*(p1[0]*p3[1] - p3[0]*p1[1]) + (p3[0]**2 + p3[1]**2)*(p2[0]*p1[1] - p1[0]*p2[1])
        if A == 0.0: # points are in a line, and circle is infinitely large
            return None
        xc = np.round(-B/(2*A)).astype(int)
        yc = np.round(-C/(2*A)).astype(int)
        R = np.round(np.sqrt((B**2 + C**2 - 4*A*D)/(4*(A**2)))).astype(int)
        return (xc, yc, R)
    
    def get_image(self, p1, p2, p3, title=None):
        img = ski.util.img_as_float(self._edge_img)
        img = ski.color.gray2rgb(img)
        xc, yc, R = ObjectiveFunc.fit_circle(p1, p2, p3)
        img[ski.draw.circle_perimeter(xc, yc, R, shape=self._edge_img.shape)] = [255, 0, 0]
        img[ski.draw.disk((p1[0], p1[1]), 5, shape=img.shape)] = [0, 255, 0]
        img[ski.draw.disk((p2[0], p2[1]), 5, shape=img.shape)] = [0, 255, 0]
        img[ski.draw.disk((p3[0], p3[1]), 5, shape=img.shape)] = [0, 255, 0]
        return img
    
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

n_imgs = 6
for idx, i in enumerate(np.random.randint(1, 1635, size=(n_imgs), dtype=int)):
    image = ski.io.imread('./images/image' + str(i) + '.jpg')
    #image = ski.io.imread('circle_test.jpg')
    obj_func = ObjectiveFunc(image)
    num_processes = cpu_count()
    num_iterations = num_processes
    results = harmony_search(obj_func, num_processes, num_iterations)
    print('Elapsed time: {}\nBest harmony: {}\nBest fitness: {}\nHarmony memories:'.format(results.elapsed_time, results.best_harmony, results.best_fitness))

    p1, p2, p3 = obj_func.solution_vec_to_points(results.best_harmony)
    circle = obj_func.get_image(p1, p2, p3)
    plt.subplot(2, n_imgs, idx+1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(2, n_imgs, idx+n_imgs+1)
    plt.imshow(circle)
    plt.axis('off')

plt.tight_layout(pad=0.1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()