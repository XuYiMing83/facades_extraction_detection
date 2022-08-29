# -*- coding: utf-8 -*-
'''
given points, find the all related filenames 
'''
import os
import numpy as np
from plyfile import PlyData, PlyElement
import pandas as pd
import struct
import pickle 
import math 
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


hildesheim_r = 25
hildesheim_x_offset = 564546
hildesheim_y_offset = 5778458

## parse file name 
def Hexa2Decimal(x):
    #tile = struct.unpack('>i', x.decode('hex'))
    tile = struct.unpack('>i', bytes.fromhex(x))
    return tile[0]

def coord(m,n,r,x_offset,y_offset):
    return r*Hexa2Decimal(m)+x_offset, r*Hexa2Decimal(n)+y_offset

def read_cellname(fn,r,x_offset,y_offset):
    m,n = fn.split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    return mm, nn

def read_fn(fn,r,x_offset,y_offset):
    m,n = fn.split('.')[0].split('_')[:2]
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    return mm, nn

def read_fn_runid(fn,r,x_offset,y_offset):
    m,n = fn.split('.')[0].split('_')
    [mm,nn] = coord(m, n, r, x_offset, y_offset)
    return mm, nn

def read_bin_double(path, skiprows=0):

    xyz = []
    with open(path, "rb") as file_:
        skip = 0
        for _ in range(skiprows):  
            skip = skip + len(next(file_))
            
        file_.seek(0, 2)
        size = file_.tell()
        file_.seek(skip, 0)
            
        while file_.tell() < size:
            binary = file_.read(8*3)
            point = struct.unpack("ddd", binary)
            xyz.append(point)

    return np.array(xyz)

def read_ply(ply_path):
    plydata = PlyData.read(ply_path)  
    data = plydata.elements[0].data  
    data_pd = pd.DataFrame(data)  
    data_np = np.zeros(data_pd.shape, dtype=np.float)  
    property_names = data[0].dtype.names  
    for i, name in enumerate(property_names):  
        data_np[:, i] = data_pd[name]
    return data_np


class Point_Find_Fn():
    
    def __init__(self, ply_root):
        self.ply_root = ply_root
        self.file_dict, self.corner_coors = self._get_ply_info()
        
    def _get_ply_info(self):
        global hildesheim_r, hildesheim_x_offset, hildesheim_y_offset
        
        file_dict = dict()
        names = os.listdir(self.ply_root)
        corner_coors = np.zeros((len(names), 2), dtype=np.int64)
        for i, fn in enumerate(names):
            mm, nn = read_fn_runid(fn,hildesheim_r,hildesheim_x_offset,hildesheim_y_offset)
            file_dict[(mm, nn)] = fn
            corner_coors[i] = np.array([mm, nn])
        inds = corner_coors.argsort(axis=0)[:,0]
        corner_coors = corner_coors[inds]
        return file_dict, corner_coors
    
    def get_fns(self, points):
        '''
        find at which blocks the all points locate.
        '''
        global hildesheim_r
        fns = list()
        for x, y in points:
            x = round(x)
            y = round(y)
            
            tmp = self.corner_coors.copy()
            tmp[:, 0] = x - tmp[:, 0]
            tmp[:, 1] = y - tmp[:, 1]
            
            ind_x1 = np.where(tmp[:, 0]>=0, 1, 0)
            ind_x2 = np.where(tmp[:, 0]<=hildesheim_r, 1, 0)
            
            ind_y1 = np.where(tmp[:, 1]>=0, 1, 0)
            ind_y2 = np.where(tmp[:, 1]<=hildesheim_r, 1, 0)
            
            ind_x = np.logical_and(ind_x1, ind_x2)
            ind_y = np.logical_and(ind_y1, ind_y2)
    
            if len(np.logical_and(ind_x, ind_y).nonzero()[0])!=0:
                ind = np.logical_and(ind_x, ind_y).nonzero()[0][0]
                fn = self.file_dict[tuple(self.corner_coors[ind])]
                fns.append(fn)
            else:
                pass
        fns = set(fns)
        return list(fns)
    
    def limit_search_space(self, points, fns, aug_range = 1.0):
        """
        return the serch space and values.
        """
        cache = list()
        for fn in fns:
            data_np = read_ply(os.path.join(self.ply_root, fn))
            cache.append(data_np)
        if len(cache) > 1:
            cache = np.concatenate(cache, axis=0)
        elif len(cache)==1:
            cache = cache[0]
        tmp_cache = cache.copy()
        x_min = points[:, 0].min() - aug_range
        x_max = points[:, 0].max() + aug_range
        y_min = points[:, 1].min() - aug_range
        y_max = points[:, 1].max() + aug_range
    
        ind_x1 = np.where(cache[:, 0]>x_min)[0]
        cache = cache[ind_x1]
        ind_x2 = np.where(cache[:, 0]<x_max)[0]
        cache = cache[ind_x2]
        ind_y1 = np.where(cache[:, 1]>y_min)[0]
        cache = cache[ind_y1]
        ind_y2 = np.where(cache[:, 1]<y_max)[0]
        cache = cache[ind_y2]
        if len(cache)<5000:
            return tmp_cache[:, :2], tmp_cache[:, 2]
        return cache[:, :2], cache[:, 2]


    
    
    
    
    
    
    
    
    
    
    
    