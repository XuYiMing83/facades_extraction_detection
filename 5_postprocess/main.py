import pickle
import numpy as np
import csv
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import random
import cv2
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure,draw
import os
from scipy import stats
import config
import math
import scipy.signal as signal
import utils
from sklearn.neighbors import KDTree
import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd
from points_find_fn import Point_Find_Fn
from lib.ply import write_points_ddd

'''
The purpose of this file is to align the scanned point cloud coordinate system to the CityGML coordinate system. 
We use topographic maps for this work.
'''


root = 'F:/ikg/3_image_generation/all/'
save_path_root = 'F:/ikg/4_detection/geometry/'
info_root = root + 'info/'
depth_dat_root = 'F:/ikg/3_image_generation/new/depth_dat/'
image_root = root + 'depth_image/'
normal_root = root + 'normal/'
txt_root = save_path_root + 'windows/final_train_predictions/txt/'
post_root = 'F:/ikg/5_CityGML/post_process/'
ply_update_value_root = 'F:/ikg/5_CityGML/20200325HildesheimV20210215/aligned_update_values/'

ply_update_values = Point_Find_Fn(ply_update_value_root)
gps2als_shift = 43.5341477257
invalid_nids = list()
plane = str()
vertex_plane = list()
plane_count = 0
plane_g = str()
vertex_plane_g = list()
plane_count_g = 0

def clear_all_windows(nid, image):
   if os.path.exists(os.path.join(txt_root, '{}.txt'.format(nid))):
      with open(os.path.join(txt_root, '{}.txt'.format(nid)), 'r') as f:
         tmp = f.readlines()
      pred = list()
      if len(tmp) == 0:
         pred = None
      else:
         pred = [np.array(list(map(float, e.split(' '))))[None, :] for e in tmp]
         pred = np.concatenate(pred)
      for e in range(len(pred)):
         pxmin, pymin, pxmax, pymax, pro = pred[e]
         image[int(pymin):int(pymax), int(pxmin):int(pxmax)] = image[0, 0]
   return image

def write_points_bin_plane(points,  plane_4, plane_count, path, fields, dtypes):
   count = len(points)
   header = """ply\nformat ascii 1.0\nelement vertex {0}\n""".format(count)
   for f, d in zip(fields, dtypes):
      header += " ".join(["property", d, f, '\n'])

   header += """property uchar red\n"""
   header += """property uchar green\n"""
   header += """property uchar blue\n"""
   header += """element face {0}\n""".format(plane_count)
   header += """property list uint8 int32 vertex_index\n"""
   header += """end_header\n"""

   with open(path, "w+") as file_:
      file_.write(header)

   with open(path, "a") as file_:
      for p in points:
         strp = str(p[0]) +'\0'+ str(p[1]) +'\0'+ str(p[2]) +'\0'+ str(p[3]) +'\0'+ str(p[4]) +'\0'+ str(p[5])+'\n'
         file_.write(strp)
   with open(path, "a") as file_:
      file_.write(str(plane_4))


def write_points_ddd_plane(points,  plane_4, plane_count, path):
   fields = ["x", "y", "z"]
   dtypes = ['double', 'double', 'double']
   write_points_bin_plane(points,  plane_4, plane_count, path, fields, dtypes)


def get_depth(pxmin, pymin, pxmax, pymax, depth_dat):
   w, l = depth_dat.shape
   temp_depth_min = depth_dat[pymin + 1:min(pymin + 16, w), max(pxmin - 15, 0):pxmin]
   mask_min = np.where(temp_depth_min == depth_dat[0, 0], 0, 1)
   temp_depth_min = np.multiply(temp_depth_min, mask_min)


   temp_depth_max = depth_dat[max(pymax - 16, 0):pymax - 1, pxmax:min(pxmax + 15, l)]
   mask_max = np.where(temp_depth_max == depth_dat[0, 0], 0, 1)
   temp_depth_max = np.multiply(temp_depth_max, mask_max)
   min_depth = 0
   max_depth = 0
   if np.sum(mask_min) != 0 and np.sum(mask_max) != 0:
       min_depth = np.sum(temp_depth_min) / np.sum(mask_min)
       max_depth = np.sum(temp_depth_max) / np.sum(mask_max)
   if np.sum(mask_min) != 0 and np.sum(mask_max) == 0:
       min_depth = np.sum(temp_depth_min) / np.sum(mask_min)
       max_depth = min_depth
   if np.sum(mask_min) == 0 and np.sum(mask_max) != 0:
       max_depth = np.sum(temp_depth_max) / np.sum(mask_max)
       min_depth = min_depth
   if abs(min_depth - max_depth) > 0.25:
       min_depth = min(min_depth, max_depth)
       max_depth = min(min_depth, max_depth)

   return min_depth, max_depth

def GML_maker(nid):
   global gps2als_shift
   global plane
   global vertex_plane
   global plane_count
   global plane_g
   global vertex_plane_g
   global plane_count_g
   window_position = list()
   geometry_position = list()
   depth_dat_path = os.path.join(depth_dat_root, '{}_new_depth.dat'.format(nid))
   info_path = os.path.join(info_root, '{}_info.dat'.format(nid))
   image_path = os.path.join(image_root, '{}.png'.format(nid))
   with open(depth_dat_path, 'rb') as f:
      depth_dat = pickle.load(f)
   depth_dat = clear_all_windows(nid, depth_dat)
   with open(info_path, 'rb') as f:
      info = pickle.load(f)
   w, l = depth_dat.shape
   matrix_back = info['trans_i2o']
   base_x = info['original_x']
   base_y = info['original_y']
   base_z = info['original_z']
   range_x_min = info['left_edge']
   range_y_min = info['buttom_edge']

   change_base_points = np.array([[(l-0.25*l) / 50 + range_x_min, 0, (w-0.25*w) / 50 + range_y_min, 1],
                                  [0.25*l / 50 + range_x_min, 0, 0.25*w / 50 + range_y_min, 1]])
   change_base_points_back = matrix_back @ change_base_points.T
   change_base_points_back = change_base_points_back.T
   change_base_points_back[:, 0] += base_x
   change_base_points_back[:, 1] += base_y
   change_base_points_back[:, 2] += base_z
   fns = ply_update_values.get_fns(change_base_points_back[:, :2])
   if len(fns) == 0:
      invalid_nids.append(nid)
      return pd.DataFrame()
   u_search_space, u_values = ply_update_values.limit_search_space(change_base_points_back[:, :2], fns)
   tree = KDTree(u_search_space, leaf_size=2)
   dists, inds = tree.query(change_base_points_back[:, :2], k=5)
   update_values = u_values[inds].mean(axis=1)
   update_values = update_values.mean()

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
         min_depth, max_depth = get_depth(int(pxmin), int(pymin), int(pxmax), int(pymax), depth_dat)
         point_p= np.array([[pxmin / 50 + range_x_min, 0.25 + min_depth, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, 0.25 + min_depth, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, 0.25 + max_depth, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, 0.25 + max_depth, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, -0.05 + min_depth, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, -0.05 + min_depth, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, -0.05 + max_depth, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, -0.05 + max_depth, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, min_depth, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmin / 50 + range_x_min, min_depth, (w - 1 - pymin) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, max_depth, (w - 1 - pymax) / 50 + range_y_min, 1],
                            [pxmax / 50 + range_x_min, max_depth, (w - 1 - pymin) / 50 + range_y_min, 1]])
         point_p_back = matrix_back @ point_p.T
         point_p_back = point_p_back.T
         point_p_back[:, 0] += base_x
         point_p_back[:, 1] += base_y
         point_p_back[:, 2] += base_z - gps2als_shift - update_values
         wp = point_p_back[8:12, 0:3].tolist()
         window_position.append(wp)

         for i_p in range(point_p_back.shape[0]):
            vertex_plane.append((point_p_back[i_p, 0], point_p_back[i_p, 1], point_p_back[i_p, 2], 0, 0, 255))
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

   geometry_dat_path = os.path.join(save_path_root, 'geometry_info', '{}.dat'.format(nid))
   with open(geometry_dat_path, 'rb') as f:
      geometry_dat = pickle.load(f)
   tem_point_g = len(vertex_plane_g)
   for i in range(len(geometry_dat)):
      xmin, ymin, xmax, ymax, z_1, z_2, z_3, z_4 = geometry_dat[i]
      mmin_depth, mmax_depth = get_depth(int(xmin), int(ymin), int(xmax), int(ymax), depth_dat)
      point_geo = np.array([[xmin / 50 + range_x_min, -0.005 + mmin_depth, (w - 1 - ymax) / 50 + range_y_min, 1],
                            [xmin / 50 + range_x_min, -0.005 + mmin_depth, (w - 1 - ymin) / 50 + range_y_min, 1],
                            [xmax / 50 + range_x_min, -0.005 + mmax_depth, (w - 1 - ymax) / 50 + range_y_min, 1],
                            [xmax / 50 + range_x_min, -0.005 + mmax_depth, (w - 1 - ymin) / 50 + range_y_min, 1],
                            [xmin / 50 + range_x_min, z_3, (w - 1 - ymax) / 50 + range_y_min, 1],
                            [xmin / 50 + range_x_min, z_1, (w - 1 - ymin) / 50 + range_y_min, 1],
                            [xmax / 50 + range_x_min, z_4, (w - 1 - ymax) / 50 + range_y_min, 1],
                            [xmax / 50 + range_x_min, z_2, (w - 1 - ymin) / 50 + range_y_min, 1]])
      point_geo_back = matrix_back @ point_geo.T
      point_geo_back = point_geo_back.T
      point_geo_back[:, 0] += base_x
      point_geo_back[:, 1] += base_y
      point_geo_back[:, 2] += base_z - gps2als_shift - update_values
      geometry_position.append(point_geo_back[:, 0:3].tolist())

      for i_geo in range(point_geo_back.shape[0]):
         vertex_plane_g.append((point_geo_back[i_geo, 0], point_geo_back[i_geo, 1], point_geo_back[i_geo, 2], 255, 0, 0))
      plane_g += '3\0' + str(tem_point_g + 8 * (i)) + '\0' + str(tem_point_g+1 + 8 * (i)) + '\0' + str(tem_point_g+2 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g+1 + 8 * (i)) + '\0' + str(tem_point_g+2 + 8 * (i)) + '\0' + str(tem_point_g+3 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g+4 + 8 * (i)) + '\0' + str(tem_point_g+5 + 8 * (i)) + '\0' + str(tem_point_g+6 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g+5 + 8 * (i)) + '\0' + str(tem_point_g+6 + 8 * (i)) + '\0' + str(tem_point_g+7 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g + 8 * (i)) + '\0' + str(tem_point_g+1 + 8 * (i)) + '\0' + str(tem_point_g+4 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g+1 + 8 * (i)) + '\0' + str(tem_point_g+4 + 8 * (i)) + '\0' + str(tem_point_g+5 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g+6 + 8 * (i)) + '\0' + str(tem_point_g+7 + 8 * (i)) + '\0' + str(tem_point_g+2 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g+7 + 8 * (i)) + '\0' + str(tem_point_g+2 + 8 * (i)) + '\0' + str(tem_point_g+3 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g+5 + 8 * (i)) + '\0' + str(tem_point_g+1 + 8 * (i)) + '\0' + str(tem_point_g+7 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g+1 + 8 * (i)) + '\0' + str(tem_point_g+7 + 8 * (i)) + '\0' + str(tem_point_g+3 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g+4 + 8 * (i)) + '\0' + str(tem_point_g + 8 * (i)) + '\0' + str(tem_point_g+6 + 8 * (i)) + '\n'
      plane_g += '3\0' + str(tem_point_g + 8 * (i)) + '\0' + str(tem_point_g+6 + 8 * (i)) + '\0' + str(tem_point_g+2 + 8 * (i)) + '\n'
      plane_count_g += 12
   dic_info = dict()
   dic_info["id"] = nid
   dic_info["facades_opening_3d"] = window_position
   dic_info["protruding_3d"] = geometry_position
   return dic_info



data_csv = list()
files = os.listdir(info_root)
for idx, file in enumerate(files):
   print('\rimage: {}/{}'.format(idx+1, len(files)), end='')
   #if idx >0:
      #break
   nid = int(file.split('_')[0])
   dic_info = GML_maker(nid)
   data_csv.append(dic_info)
data_csv = sorted(data_csv, key=lambda items: items['id'])
header = ["id", "facades_opening_3d", "protruding_3d"]
with open(r'E:\datacsv\3d_data.csv', 'w', newline='') as f:
   dictWriter = csv.DictWriter(f, header)
   dictWriter.writeheader()
   dictWriter.writerows(data_csv)
with open(r'E:\datacsv\3d_data.dat', 'wb') as f:
   pickle.dump(data_csv, f)
write_points_ddd_plane(vertex_plane, plane, plane_count, os.path.join(post_root, 'window_plane.ply'))
write_points_ddd_plane(vertex_plane_g, plane_g, plane_count_g, os.path.join(post_root, 'geo_plane.ply'))

