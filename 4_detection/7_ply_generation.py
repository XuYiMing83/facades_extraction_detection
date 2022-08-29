# -*- coding: utf-8 -*-
import numpy as np
import struct
import pickle
from collections import Counter
from skimage import io
import cv2
import matplotlib.pyplot as plt
#from tqdm import tqdm
import itertools
import time
from numba import jit
import math
from PIL import Image, ImageDraw, ImageFont
import config
import os
import csv
import utils
import os
from lib.ply import write_points_ddd
from plyfile import PlyData, PlyElement

import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
root = 'F:/ikg/3_image_generation/all/'
save_path_root = 'F:/ikg/4_detection/geometry/'
info_root = root + 'info/'
depth_dat_root = root + 'post_depth_dat/'
image_root = root + 'depth_image/'
normal_root = root + 'normal/'
txt_root = save_path_root + 'windows/final_train_predictions/txt/'
if not os.path.exists(save_path_root):
   os.makedirs(save_path_root)

def write_points_bin_plane(points,  plane_4, plane_count, edge, edge_count, path, fields, dtypes):
   count = len(points)
   header = """ply\nformat ascii 1.0\nelement vertex {0}\n""".format(count)
   for f, d in zip(fields, dtypes):
      header += " ".join(["property", d, f, '\n'])

   header += """property uchar red\n"""
   header += """property uchar green\n"""
   header += """property uchar blue\n"""
   header += """element face {0}\n""".format(plane_count)
   header += """property list uint8 int32 vertex_index\n"""
   header += """element edge {0}\n""".format(edge_count)
   header += """property int32 vertex1\n"""
   header += """property int32 vertex2\n"""
   header += """end_header\n"""

   with open(path, "w+") as file_:
      file_.write(header)

   with open(path, "a") as file_:
      for p in points:
         strp = str(p[0]) +'\0'+ str(p[1]) +'\0'+ str(p[2]) +'\0'+ str(p[3]) +'\0'+ str(p[4]) +'\0'+ str(p[5])+'\n'
         file_.write(strp)
   with open(path, "a") as file_:
      file_.write(str(plane_4))
   with open(path, "a") as file_:
      file_.write(str(edge))


def write_points_ddd_plane(points,  plane_4, plane_count, edge, edge_count, path):
   fields = ["x", "y", "z"]
   dtypes = ['double', 'double', 'double']
   write_points_bin_plane(points,  plane_4, plane_count, edge, edge_count, path, fields, dtypes)

def cloud_point_maker(nid):
   depth_dat_path = os.path.join(depth_dat_root, '{}_depth.dat'.format(nid))
   info_path = os.path.join(info_root, '{}_info.dat'.format(nid))
   image_path = os.path.join(image_root, '{}.png'.format(nid))
   with open(depth_dat_path, 'rb') as f:
      depth_dat = pickle.load(f)
   with open(info_path, 'rb') as f:
      info = pickle.load(f)
   matrix_back = info['trans_i2o']
   base_x = info['original_x']
   base_y = info['original_y']
   base_z = info['original_z']
   range_x_min = info['left_edge']
   range_y_min = info['buttom_edge']
   w, l = depth_dat.shape
   points_back = list()
   for i in range(w):
      for j in range(l):
         if depth_dat[i, j] != depth_dat[0, 0]:
            point_in = np.array([j / 50 + range_x_min, depth_dat[i, j], (w - i - 1) / 50 + range_y_min, 1])
            point_back = matrix_back @ point_in.T
            points_back.append((point_back[0]+base_x, point_back[1]+base_y, point_back[2]+base_z))
   ply_save_path = save_path_root + 'ply_cloud'
   if not os.path.exists(ply_save_path):
      os.makedirs(ply_save_path)
   write_points_ddd(points_back, os.path.join(ply_save_path, '{}.ply'.format(nid)))
   image_depth = cv2.imread(image_path, 0)
   x_min = np.min(np.nonzero(image_depth)[1])
   x_max = np.max(np.nonzero(image_depth)[1])
   y_min = np.max(np.nonzero(image_depth)[0])
   y_max = np.min(np.nonzero(image_depth)[0])
   point_plane = np.array([[x_min / 50 + range_x_min, 0, (w - 1 - y_min) / 50 + range_y_min, 1],
                            [x_min / 50 + range_x_min, 0, (w - 1 - y_max) / 50 + range_y_min, 1],
                            [x_max / 50 + range_x_min, 0, (w - 1 - y_min) / 50 + range_y_min, 1],
                            [x_max / 50 + range_x_min, 0, (w - 1 - y_max) / 50 + range_y_min, 1]])
   point_plane_back = matrix_back @ point_plane.T
   point_plane_back = point_plane_back.T
   point_plane_back[:, 0] += base_x
   point_plane_back[:, 1] += base_y
   point_plane_back[:, 2] += base_z
   vertex_plane = list()
   edge = str()
   for i in range(point_plane_back.shape[0]):
      vertex_plane.append((point_plane_back[i, 0], point_plane_back[i, 1], point_plane_back[i, 2], 0, 0, 255))
   plane = '3 0 1 2\n3 1 2 3\n'
   plane_count = 2
   geometry_dat_path = os.path.join(save_path_root,'geometry_info', '{}.dat'.format(nid))
   with open(geometry_dat_path, 'rb') as f:
      geometry_dat = pickle.load(f)
   for i in range(len(geometry_dat)):
      xmin, ymin, xmax, ymax, z_1, z_2, z_3, z_4 = geometry_dat[i]
      point_geo = np.array([[xmin / 50 + range_x_min, -0.005, (w - 1 - ymax) / 50 + range_y_min, 1],
                            [xmin / 50 + range_x_min, -0.005, (w - 1 - ymin) / 50 + range_y_min, 1],
                            [xmax / 50 + range_x_min, -0.005, (w - 1 - ymax) / 50 + range_y_min, 1],
                            [xmax / 50 + range_x_min, -0.005, (w - 1 - ymin) / 50 + range_y_min, 1],
                            [xmin / 50 + range_x_min, z_3, (w - 1 - ymax) / 50 + range_y_min, 1],
                            [xmin / 50 + range_x_min, z_1, (w - 1 - ymin) / 50 + range_y_min, 1],
                            [xmax / 50 + range_x_min, z_4, (w - 1 - ymax) / 50 + range_y_min, 1],
                            [xmax / 50 + range_x_min, z_2, (w - 1 - ymin) / 50 + range_y_min, 1]])
      point_geo_back = matrix_back @ point_geo.T
      point_geo_back = point_geo_back.T
      point_geo_back[:, 0] += base_x
      point_geo_back[:, 1] += base_y
      point_geo_back[:, 2] += base_z
      for i_geo in range(point_geo_back.shape[0]):
         vertex_plane.append((point_geo_back[i_geo, 0], point_geo_back[i_geo, 1], point_geo_back[i_geo, 2], 255, 0, 0))
      plane += '3\0' + str(4 + 8 * (i)) + '\0' + str(5 + 8 * (i)) + '\0' + str(6 + 8 * (i)) + '\n'
      plane += '3\0' + str(5 + 8 * (i)) + '\0' + str(6 + 8 * (i)) + '\0' + str(7 + 8 * (i)) + '\n'
      plane += '3\0' + str(8 + 8 * (i)) + '\0' + str(9 + 8 * (i)) + '\0' + str(10 + 8 * (i)) + '\n'
      plane += '3\0' + str(9 + 8 * (i)) + '\0' + str(10 + 8 * (i)) + '\0' + str(11 + 8 * (i)) + '\n'
      plane += '3\0' + str(4 + 8 * (i)) + '\0' + str(5 + 8 * (i)) + '\0' + str(8 + 8 * (i)) + '\n'
      plane += '3\0' + str(5 + 8 * (i)) + '\0' + str(8 + 8 * (i)) + '\0' + str(9 + 8 * (i)) + '\n'
      plane += '3\0' + str(10 + 8 * (i)) + '\0' + str(11 + 8 * (i)) + '\0' + str(6 + 8 * (i)) + '\n'
      plane += '3\0' + str(11 + 8 * (i)) + '\0' + str(6 + 8 * (i)) + '\0' + str(7 + 8 * (i)) + '\n'
      plane += '3\0' + str(9 + 8 * (i)) + '\0' + str(5 + 8 * (i)) + '\0' + str(11 + 8 * (i)) + '\n'
      plane += '3\0' + str(5 + 8 * (i)) + '\0' + str(11 + 8 * (i)) + '\0' + str(7 + 8 * (i)) + '\n'
      plane += '3\0' + str(8 + 8 * (i)) + '\0' + str(4 + 8 * (i)) + '\0' + str(10 + 8 * (i)) + '\n'
      plane += '3\0' + str(4 + 8 * (i)) + '\0' + str(10 + 8 * (i)) + '\0' + str(6 + 8 * (i)) + '\n'
      plane_count += 12
   #write_points_ddd_plane(vertex_plane, plane, plane_count, os.path.join(ply_save_path, '{}_plane.ply'.format(nid)))
   edge_count = 0
   if os.path.exists(os.path.join(txt_root, '{}.txt'.format(nid))):
      with open(os.path.join(txt_root, '{}.txt'.format(nid)), 'r') as f:
         tmp = f.readlines()
      pred = list()
      if len(tmp) == 0:
         pred = None
      else:
         pred = [np.array(list(map(float, e.split(' '))))[None, :] for e in tmp]
         pred = np.concatenate(pred)
      tem_point = len(vertex_plane)
      for e in range(len(pred)):
         pxmin, pymin, pxmax, pymax, pro = pred[e]
         point_p= np.array([
                            [pxmin / 50 + range_x_min, 0.02, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, 0.02, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, 0.02, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, 0.02, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, -0.02, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, -0.02, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, -0.02, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, -0.02, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, 0, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, 0, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, 0, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, 0, (w - 1 - pymin) / 50 + range_y_min, 1]])
         point_p_back = matrix_back @ point_p.T
         point_p_back = point_p_back.T
         point_p_back[:, 0] += base_x
         point_p_back[:, 1] += base_y
         point_p_back[:, 2] += base_z
         for i_p in range(point_p_back.shape[0]):
            vertex_plane.append((point_p_back[i_p, 0], point_p_back[i_p, 1], point_p_back[i_p, 2], 0, 255, 0))
         plane += '3\0' + str(tem_point + 12 * (e)) + '\0' + str(tem_point+1 + 12 * (e)) + '\0' + str(tem_point+2 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point+1 + 12 * (e)) + '\0' + str(tem_point+2 + 12 * (e)) + '\0' + str(tem_point+3 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point+4 + 12 * (e)) + '\0' + str(tem_point+5 + 12 * (e)) + '\0' + str(tem_point+6 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point+5 + 12 * (e)) + '\0' + str(tem_point+6 + 12 * (e)) + '\0' + str(tem_point+7 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point + 12 * (e)) + '\0' + str(tem_point+1 + 12 * (e)) + '\0' + str(tem_point+4 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point+1 + 12 * (e)) + '\0' + str(tem_point+4 + 12 * (e)) + '\0' + str(tem_point+5 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point+6 + 12 * (e)) + '\0' + str(tem_point+7 + 12 * (e)) + '\0' + str(tem_point+2 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point+7 + 12 * (e)) + '\0' + str(tem_point+2 + 12 * (e)) + '\0' + str(tem_point+3 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point+5 + 12 * (e)) + '\0' + str(tem_point+1 + 12 * (e)) + '\0' + str(tem_point+7 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point+1 + 12 * (e)) + '\0' + str(tem_point+7 + 12 * (e)) + '\0' + str(tem_point+3 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point+4 + 12 * (e)) + '\0' + str(tem_point + 12 * (e)) + '\0' + str(tem_point+6 + 12 * (e)) + '\n'
         plane += '3\0' + str(tem_point + 12 * (e)) + '\0' + str(tem_point+6 + 12 * (e)) + '\0' + str(tem_point+2 + 12 * (e)) + '\n'
         plane_count += 12
         edge += str(tem_point + 8 + 4 * (e)) + '\0' + str(tem_point + 9 + 4 * (e)) + '\n'
         edge += str(tem_point + 8 + 4 * (e)) + '\0' + str(tem_point + 10 + 4 * (e)) + '\n'
         edge += str(tem_point + 9 + 4 * (e)) + '\0' + str(tem_point + 11 + 4 * (e)) + '\n'
         edge += str(tem_point + 10 + 4 * (e)) + '\0' + str(tem_point + 11 + 4 * (e)) + '\n'
         edge_count += 4
   plane_save_path = save_path_root + 'ply_plane'
   if not os.path.exists(plane_save_path):
      os.makedirs(plane_save_path)
   write_points_ddd_plane(vertex_plane, plane, plane_count, edge, edge_count, os.path.join(plane_save_path, '{}_plane.ply'.format(nid)))



files = os.listdir(info_root)
for idx, file in enumerate(files):
   print('\rimage: {}/{}'.format(idx+1, len(files)), end='')
   nid = int(file.split('_')[0])
   cloud_point_maker(nid)