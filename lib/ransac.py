# -*- coding: utf-8 -*-

from skimage import io
import pickle
import numpy as np
import math
from PIL import Image, ImageDraw                  #导入Image 类
import matplotlib.pyplot as plt

from sklearn import linear_model
import pcl
import random



class RANSAC():
    
    def __init__(self):
        pass
    def augment(self, xyzs):
        axyz = np.ones((len(xyzs), 4))
        axyz[:, :3] = xyzs
        return axyz

    def estimate(self, xyzs):
        axyz = self.augment(xyzs[:3])
        return np.linalg.svd(axyz)[-1][-1, :]
    
    def is_inlier(self, coeffs, xyz, threshold):
        a,b,c,d = coeffs
        x, y, z = xyz
        return abs(a*x + b*y +c*z + d)/math.sqrt(a**2+b**2+c**2) < threshold
    
    def run_ransac(self, data, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
        best_ic = 0
        best_model = None
        best_idx = None
        random.seed(random_seed)
        # random.sample cannot deal with "data" being a numpy array
        data = list(data)
        inliner_mask = list()
        run_times = 0
        for i in range(max_iterations):
            print('\rransac: {}/{}'.format(i, max_iterations), end='')
            run_times += 1
            s = random.sample(data, int(sample_size))
            m = self.estimate(s)
            ic = 0
            tmp = list()
            for j in range(len(data)):
                if is_inlier(m, data[j]):
                    ic += 1
                    tmp.append(j)
    
            if ic > best_ic:
                best_ic = ic
                best_model = m
                best_idx = np.array(tmp)
                if ic > goal_inliers and stop_at_goal:
                    break
        print(" ")
        #print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
        return best_model, best_idx, run_times
    
    def run(self, xyzs, inlier_thres = 0.7, max_iterations=50, threshold=0.075):
        #print(max_iterations)
        n = len(xyzs)
        goal_inliers = n * inlier_thres
    
        # RANSAC
        m, mask, run_times = self.run_ransac(xyzs, lambda x, y: self.is_inlier(x, y, threshold), 30, goal_inliers, max_iterations)
        a, b, c, d = m
        
        return m, mask, run_times < max_iterations
