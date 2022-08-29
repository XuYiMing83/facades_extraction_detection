import xlrd
import sys
sys.path.insert(0, '../all_segmetations')
import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd
import geojson
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from tqdm import tqdm
import pickle
import math
import config
from skimage import io
from sklearn import linear_model
import random
import cv2
import lib.ransac as ransac
root = 'F:/ikg/3_image_generation/all/'
save_path_root = 'F:/ikg/4_detection/geometry/'
info_root = root + 'info/'
depth_dat_root = root + 'post_depth_dat/'
image_root = root + 'post_depth_image/'
depth_dat_root_new = 'F:/ikg/3_image_generation/new/depth_dat/'
normal_root = root + 'normal/'
normal_info_root = root + 'normal_info/'
if not os.path.exists(save_path_root):
   os.makedirs(save_path_root)


def divide_geometry(nid):
    info_path = os.path.join(info_root, '{}_info.dat'.format(nid))
    image_path = os.path.join(image_root, '{}.png'.format(nid))
    depth_dat_path = os.path.join(depth_dat_root, '{}_depth.dat'.format(nid))
    depth_dat_new_path = os.path.join(depth_dat_root_new, '{}_new_depth.dat'.format(nid))
    normal_path = os.path.join(normal_root, '{}.png'.format(nid))
    normal_info_path = os.path.join(normal_info_root, '{}.dat'.format(nid))
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    with open(depth_dat_path, 'rb') as f:
        depth_dat = pickle.load(f)
    with open(depth_dat_new_path, 'rb') as f:
        depth_dat_new = pickle.load(f)
    with open(normal_info_path, 'rb') as f:
        norm = pickle.load(f)
    max_in_new = np.max(depth_dat_new)
    #norm = cv2.imread(normal_path, 1)
    #norm = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)
    image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facade_area = info['facade_area']
    ref_plane = info['ref_plane']
    cover_mask = np.ones(depth_dat.shape)

    norm_x = norm[:, :, 0]
    norm_y = norm[:, :, 1]
    norm_xy = np.sqrt(norm_x ** 2 + norm_y ** 2)
    cover_mask = np.where(norm_xy < 0.65, -1, cover_mask)
    #image = np.where(cover_mask == -1, image[0, 0], image)
    #io.imsave(os.path.join('E:\lunwentupian', '{}.png'.format(nid)), image)
    #raise NotImplementedError()
    cover_mask = np.where(depth_dat == depth_dat[0, 0], -1, cover_mask)
    cover_mask = np.where(depth_dat > max_in_new+3.6, -1, cover_mask)
    confirm_mask = cover_mask.copy()
    count_image = np.where(cover_mask == -1, 0, 1)
    count_point = np.sum(count_image)
    data_ransac = list()
    geometry = list()
    model = ransac.RANSAC()
    '''
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if cover_mask[i, j] != -1:
                data_ransac.append((i, j, depth_dat[i, j]))
    data_ransac = np.array(data_ransac)
    #print(data_ransac)
    m, inlier_mask, _ = model.run(data_ransac, inlier_thres=0.6, max_iterations=50, threshold=0.1)
    inliers = data_ransac[inlier_mask]
    #inlier_image = np.zeros(image.shape)
    for i in range(len(inliers)):
        #inlier_image[inliers[i][0], inliers[i][1]] = inliers[i][2]
        cover_mask[int(inliers[i][0]), int(inliers[i][1])] = -1
    comp_image = np.zeros(depth_dat.shape)
    a, b, c, d = m
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if cover_mask[i, j] != -1:
                comp_image[i, j] = (a * i + b * j + d)/(-c)
    #print('comp_image',np.mean(comp_image),ref_plane)
    cover_mask = np.where(depth_dat > comp_image + 0.25, cover_mask, -1)
    '''
    #print(ref_plane)
    cover_mask = np.where(depth_dat > 0.4, cover_mask, -1)
    count_image = np.where(cover_mask == -1, 0, 1)
    re_point = np.sum(count_image)
    kernel = np.ones((3, 3))
    t = 0
    while re_point>50:
        t += 1
        data_ransac = list()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if cover_mask[i, j] != -1:
                    data_ransac.append((i, j, depth_dat[i, j]))
        data_ransac = np.array(data_ransac)
        if len(data_ransac) < 55:
            break
        mi, inlier_maski, _ = model.run(xyzs=data_ransac, inlier_thres=0.50, max_iterations=100, threshold=0.025)
        ai, bi, ci, di = mi
        inliersi = list(data_ransac[inlier_maski])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if cover_mask[i, j] != -1:
                    z_temp = (ai * i + bi * j + di) / (-ci)
                    if abs(depth_dat[i, j] - z_temp) < 0.15:
                        inliersi.append((i, j, depth_dat[i, j]))
        inlier_imagei = np.zeros(depth_dat.shape)
        cover_mask_notchange = cover_mask.copy()
        for i in range(len(inliersi)):
            inlier_imagei[int(inliersi[i][0]), int(inliersi[i][1])] = 255
            cover_mask[int(inliersi[i][0]), int(inliersi[i][1])] = -1
        if abs(ci / math.sqrt(ai ** 2 + bi ** 2 + ci ** 2)) < 0.85:
            continue
        inlier_imagei = cv2.dilate(inlier_imagei, kernel, iterations=7).astype(np.uint8)
        inlier_imagei = cv2.erode(inlier_imagei, kernel, iterations=10).astype(np.uint8)
        inlier_imagei = cv2.dilate(inlier_imagei, kernel, iterations=3).astype(np.uint8)
        contours, h = cv2.findContours(inlier_imagei, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            for i in range(len(contours)):
                xmax = max(contours[i][:, 0, 0])
                xmin = min(contours[i][:, 0, 0])
                ymax = max(contours[i][:, 0, 1])
                ymin = min(contours[i][:, 0, 1])
                A_by = (abs(ymax - ymin) * abs(xmax - xmin)) / 2500
                #print('A_by', A_by)
                copy_norm = norm[ymin+int((ymax-ymin)/4):ymax-int((ymax-ymin)/4), xmin+int((xmax-xmin)/4):xmax-int((xmax-xmin)/4), :].copy()
                copy_cover_mask = cover_mask_notchange[ymin+int((ymax-ymin)/4):ymax-int((ymax-ymin)/4), xmin+int((xmax-xmin)/4):xmax-int((xmax-xmin)/4)].copy()
                copy_norm_usable = list()
                for cni in range(copy_norm.shape[0]):
                    for cnj in range(copy_norm.shape[1]):
                        if copy_cover_mask[cni, cnj] != -1:
                            copy_norm_usable.append(copy_norm[cni, cnj])
                copy_norm_usable = np.array(copy_norm_usable)
                var_x = 0.3
                var_y = 0.3
                var_z = 0.3
                if len(copy_norm_usable) != 0:
                    var_x = np.var(copy_norm_usable[:, 0])
                    var_y = np.var(copy_norm_usable[:, 1])
                    var_z = np.var(copy_norm_usable[:, 2])


                if A_by >= 2.4 and ymin < depth_dat.shape[0]-120:
                    cont = list()
                    cont.append(contours[i])
                    by_m = np.zeros(depth_dat.shape)
                    by_m = cv2.drawContours(by_m, cont, -1, color=1, thickness=-1)

                    A_by_m = np.sum(by_m)
                    C_by_m = len(contours[i])
                    compactness = C_by_m ** 2 / (4 * math.pi * A_by_m)
                    density = (A_by_m/2500)/A_by

                    if compactness >= 1.27324:
                        z_1 = (ai * ymin + bi * xmin + di) / (-ci)
                        z_2 = (ai * ymin + bi * xmax + di) / (-ci)
                        z_3 = (ai * ymax + bi * xmin + di) / (-ci)
                        z_4 = (ai * ymax + bi * xmax + di) / (-ci)
                        z_mean = (z_1 + z_2 + z_3 + z_4)/4
                        #print(z_mean, compactness, max_in_new+0.7)
                        if density>0.8:
                            var_max = 0.8
                        else:
                            var_max = 0.5
                        if ((z_mean > max_in_new+0.7 and compactness < 6.5) or z_mean < max_in_new + 0.7) and\
                                var_x + var_y + var_z < var_max and density>0.25:
                            geometry.append([xmin, ymin, xmax, ymax, z_1, z_2, z_3, z_4])
                            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                        #cover_mask[ymin:ymax,xmin:xmax] = -1
        #comp_imagei = np.zeros(depth_dat.shape)
        #for i in range(image.shape[0]):
            #for j in range(image.shape[1]):
                #if cover_mask[i, j] != -1:
                    #comp_imagei[i, j] = (ai * i + bi * j + di) / (-ci)
        #cover_mask = np.where(depth_dat > comp_imagei-0.15, cover_mask, -1)
        count_image = np.where(cover_mask == -1, 0, 1)
        re_point = np.sum(count_image)
        #print(re_point)
        if re_point / count_point <= 0.005 or t > 3 or re_point <= 55:
            break
    image_save_path = save_path_root + 'image_rectangle'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    io.imsave(os.path.join(image_save_path, '{}.png'.format(nid)), image)
    balc_info_path = save_path_root + 'geometry_info'
    if not os.path.exists(balc_info_path):
        os.makedirs(balc_info_path)
    with open(os.path.join(balc_info_path, '{}.dat'.format(nid)), 'wb') as f:
        pickle.dump(geometry, f)
    balc_txt_path = save_path_root + 'geometry_txt'
    if not os.path.exists(balc_txt_path):
        os.makedirs(balc_txt_path)
    with open(os.path.join(balc_txt_path, '{}_info.txt'.format(nid)), 'w') as file_handle:
        for i in range(len(geometry)):
            file_handle.write(str(geometry[i]))
            file_handle.write('\n')





files = os.listdir(info_root)
for idx, file in enumerate(files):
   nid = int(file.split('_')[0])
   print('{} / {} ...nid={}'.format(idx+1, len(files), nid))

   divide_geometry(nid)




