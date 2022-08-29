# -*- coding: utf-8 -*-
import numpy as np
import math


class Find_inlier():
    
    def __init__(self):
        pass
    
    def is_inlier(self, coeffs, xyz, threshold):
        a,b,c,d = coeffs
        x, y, z = xyz
        return abs(a*x + b*y +c*z + d)/math.sqrt(a**2+b**2+c**2) < threshold
    
    def run_ransac(self, data, m, is_inlier):
        data = list(data)

        tmp = list()
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                tmp.append(j)

        idx = np.array(tmp)
        print(" ")
        return idx
    
    def run(self, xyzs, m):
        mask = self.run_ransac(xyzs, m, lambda x, y: self.is_inlier(x, y, 0.075))
        
        return mask
