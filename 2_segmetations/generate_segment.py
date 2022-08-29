# -*- coding: utf-8 -*-
import numpy as np
import struct
import pickle
from collections import Counter    
from skimage import io
import cv2
import matplotlib.pyplot as plt
#from tqdm import tqdm
from region_growing import RegionGrowing
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

import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd

import warnings
warnings.filterwarnings('ignore')



def load_file(name, path):
    with open(os.path.join(path, "{}.dat".format(name)), 'rb') as f:
        tmp = pickle.load(f)
    return tmp


def save_file(data, path):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
    

def extract_scene(path):
    '''
    extract the objects which perpendicular to ground
    '''
    norm = load_file('normal', path)
    
    norm_x = norm[:,:,0]
    norm_y = norm[:,:,1]
    norm_xy = np.sqrt(norm_x**2+norm_y**2)
    
    norm_xy = np.expand_dims(norm_xy, 2).repeat(3, axis=2)
    building = np.where(norm_xy > config.normal_vector_facade, norm, 0)
    return building

def generate_depth_image(path):
    coor = load_file('coordinate', path)
    head = load_file('head', path)

    tmp = coor - head
    tmp = tmp**2
    tmp = tmp.sum(axis=2)
    dis = np.sqrt(tmp)
    return dis




def slope(p1, p2):
    return (p1[1]-p2[1])/(p1[0]-p2[0])    


def Count_the_Points(seg_building,path):
    seg_building = np.where(seg_building == -1, 0, 1)
    sum = np.sum(seg_building)
    print(sum)
    with open(os.path.join(path, "Count_the_Points.txt"), "w") as f:
        f.write(str(sum))


def separate_high_Wall(seg, bounding, nid):
    x_min = bounding[nid][1]['x_min'].x
    y_min = bounding[nid][1]['y_min'].y
    x_max = bounding[nid][1]['x_max'].x
    y_max = bounding[nid][1]['y_max'].y
    bounding_seg = np.zeros(((x_max-x_min), (y_max-y_min)), np.uint8)
    bounding_seg_full = np.zeros(seg.shape)
    for i in range(x_max-x_min):
        for j in range(y_max - y_min):
            if seg[x_min+i][y_min+j] == nid:
                bounding_seg[i][j] = 255
    #cv2.imshow('bounding_seg_not', bounding_seg)
    kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]], np.uint8)
    bounding_seg = cv2.dilate(bounding_seg, kernel)
    bounding_seg = cv2.erode(bounding_seg, kernel)
    contours, h = cv2.findContours(bounding_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        bounding_seg_full = cv2.drawContours(bounding_seg_full, contours, -1, color=255, thickness=-1)
    #cv2.imshow('bounding_seg',bounding_seg)
    #cv2.imshow('bounding_seg_full',bounding_seg_full)
    #cv2.waitKey(0)
    bounding_seg[bounding_seg == 255] = 1
    bounding_seg_full[bounding_seg_full == 255] = 1
    sum_bounding_seg = np.sum(bounding_seg)
    sum_bounding_seg_full = np.sum(bounding_seg_full)
    full_rate = sum_bounding_seg/sum_bounding_seg_full

    return full_rate


def region_merge(seg, bounding,path, search_range, split_long_seg=False):
    coor = load_file('coordinate', path)

    rows, columns = seg.shape[0:2]
    search_mask = set()
    for idx_r in range(rows):
        print("\rregion merging: {:.2f}%".format(100 * idx_r / float(rows)), end='')

        for idx_c in range(columns):
            cur_id = seg[idx_r, idx_c]
            if cur_id == -1 or cur_id in search_mask:
                continue
            cur_norm, cur_bounding, cur_bounding_3d, cur_depth, cur_ref = bounding[cur_id]

            y_max = cur_bounding['y_max'].y
            y_min = cur_bounding['y_min'].y
            x_max = cur_bounding['x_max'].x
            x_min = cur_bounding['x_min'].x
            range_seg = seg[max(0, x_min - search_range):min(rows, x_max + search_range),
                        max(0, y_min - search_range):min(columns, y_max + search_range)]

            new_seg_id = set(range_seg.reshape(-1))
            for nid in new_seg_id:
                if nid == -1 or nid == cur_id:
                    continue
                v1 = cur_norm
                v2 = bounding[nid][0]
                x_min3d = bounding[nid][2]['x_min']
                y_min3d = bounding[nid][2]['y_min']
                x_max3d = bounding[nid][2]['x_max']
                y_max3d = bounding[nid][2]['y_max']
                x_min_n = bounding[nid][1]['x_min'].x
                y_min_n = bounding[nid][1]['y_min'].y
                x_max_n = bounding[nid][1]['x_max'].x
                y_max_n = bounding[nid][1]['y_max'].y

                similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2)))

                nid_seg = np.where(range_seg == nid, 1, 0)
                sum_nid_points = np.sum(nid_seg)
                if x_min_n > x_min and x_max_n < x_max and y_min_n > y_min and y_max_n < y_max and sum_nid_points < config.sum_nid_points:
                    a = 0.2
                else:
                    a = 0.95

                if similarity > a:

                    n_3d = bounding[nid][2]

                    d = abs(cur_depth - bounding[nid][3])
                    d_lines = utils.l2_distance_lines(cur_bounding_3d['y_min'], cur_bounding_3d['y_max'],
                                                      n_3d['y_min'], n_3d['y_max'])

                    cur_x = 0
                    cur_y = 0
                    sum_xy_in_this_area = 0
                    for i in range(max(x_min_n-5,0), min(x_max_n+5,3000)):
                        for j in range(y_min_n,y_max_n):
                            if seg[i, j] == cur_id:
                                cur_x += coor[i, j][0]
                                cur_y += coor[i, j][1]
                                sum_xy_in_this_area += 1
                    if sum_xy_in_this_area >= 1:
                        cur_x = cur_x / sum_xy_in_this_area
                        cur_y = cur_y / sum_xy_in_this_area

                    cur_x_n = 0
                    cur_y_n = 0
                    sum_xy_in_this_area_n = 0
                    for i in range(x_min_n, x_max_n):
                        for j in range(y_min_n,y_max_n):
                            if seg[i, j] == nid:
                                cur_x_n += coor[i, j][0]
                                cur_y_n += coor[i, j][1]
                                sum_xy_in_this_area_n += 1
                    if sum_xy_in_this_area_n >= 1:
                        cur_x_n = cur_x_n / sum_xy_in_this_area_n
                        cur_y_n = cur_y_n / sum_xy_in_this_area_n
                    dis_middle = np.sqrt((cur_x-cur_x_n)**2+(cur_y-cur_y_n)**2)

                    ymin_coor, ymax_coor, cover_rate, _ = utils.line2line_project(bounding[cur_id][2]['y_min'][:2],
                                                                                  bounding[cur_id][2]['y_max'][:2],
                                                                                  bounding[nid][2]['y_min'][:2],
                                                                                  bounding[nid][2]['y_max'][:2])
                    breaked_facade = abs(max(x_min3d[2], x_max3d[2])-min(cur_bounding_3d['x_min'][2],cur_bounding_3d[
                        'x_max'][2])) < 2 and dis_middle <= 0.7 #and (parallel_d < 2 and parallel_d != 0))


                    ref_z = np.mean(coor[rows // 2, cur_bounding['y_min'].y:cur_bounding['y_max'].y, 2])
                    big_balcony = d < 3.6 and cover_rate < 0.8 and abs(min(x_min3d[2], x_max3d[2]) - ref_z) > 2.3

                    if not split_long_seg:

                        if (cover_rate < 0.7 and d_lines < 2.5) or \
                                (cover_rate > 0.7 and cover_rate < 0.95 and d < config.threshold_dis * (
                                        1 + 0.1 * search_range / float(config.search_range)) and abs(
                                    cur_ref - bounding[nid][4]) < 0.1 and d_lines < 0.5) or \
                                breaked_facade or big_balcony:
                            if bounding[nid][1]['x_max'].x > bounding[cur_id][1]['x_max'].x:
                                bounding[cur_id][1]['x_max'] = bounding[nid][1]['x_max']
                                bounding[cur_id][2]['x_max'] = bounding[nid][2]['x_max']
                            if bounding[nid][1]['y_max'].y > bounding[cur_id][1]['y_max'].y:
                                bounding[cur_id][1]['y_max'] = bounding[nid][1]['y_max']
                                # bounding[cur_id][2]['y_max'] = bounding[nid][2]['y_max']
                            if bounding[nid][1]['x_min'].x < bounding[cur_id][1]['x_min'].x:
                                bounding[cur_id][1]['x_min'] = bounding[nid][1]['x_min']
                                bounding[cur_id][2]['x_min'] = bounding[nid][2]['x_min']
                            if bounding[nid][1]['y_min'].y < bounding[cur_id][1]['y_min'].y:
                                bounding[cur_id][1]['y_min'] = bounding[nid][1]['y_min']
                                # bounding[cur_id][2]['y_min'] = bounding[nid][2]['y_min']
                            bounding[cur_id][2]['y_min'] = ymin_coor
                            bounding[cur_id][2]['y_max'] = ymax_coor
                            seg = np.where(seg == nid, cur_id, seg)
                    else:
                        if (cover_rate < 0.7 and d_lines < 2.5) or breaked_facade or big_balcony:
                            if bounding[nid][1]['x_max'].x > bounding[cur_id][1]['x_max'].x:
                                bounding[cur_id][1]['x_max'] = bounding[nid][1]['x_max']
                                bounding[cur_id][2]['x_max'] = bounding[nid][2]['x_max']
                            if bounding[nid][1]['y_max'].y > bounding[cur_id][1]['y_max'].y:
                                bounding[cur_id][1]['y_max'] = bounding[nid][1]['y_max']
                                # bounding[cur_id][2]['y_max'] = bounding[nid][2]['y_max']
                            if bounding[nid][1]['x_min'].x < bounding[cur_id][1]['x_min'].x:
                                bounding[cur_id][1]['x_min'] = bounding[nid][1]['x_min']
                                bounding[cur_id][2]['x_min'] = bounding[nid][2]['x_min']
                            if bounding[nid][1]['y_min'].y < bounding[cur_id][1]['y_min'].y:
                                bounding[cur_id][1]['y_min'] = bounding[nid][1]['y_min']
                                # bounding[cur_id][2]['y_min'] = bounding[nid][2]['y_min']
                            bounding[cur_id][2]['y_min'] = ymin_coor
                            bounding[cur_id][2]['y_max'] = ymax_coor
                            seg = np.where(seg == nid, cur_id, seg)

            search_mask.add(cur_id)

    return seg, bounding



def split_building_fence(path, save_path):
    '''
    separate the facades and fences/cars...
    '''
    seg, bounding = load_file("tmp_regiongmerge", save_path)  # avg_norm, bounding_2d, bounding_3d, avg_depth, avg_ref
    coor = load_file("coordinate", path)
    rows, columns = seg.shape
    bounding_fence = dict()
    bounding_building = dict()
    seg_fence = -np.ones(seg.shape)
    seg_building = -np.ones(seg.shape)
    fence_id = 0
    building_id = 0
    print("\rsplit building fence...", end='')

    full_rates=list()

    for nid in set(seg.reshape(-1)):
        if nid == -1:
            continue

        x_min3d = bounding[nid][2]['x_min']
        y_min3d = bounding[nid][2]['y_min']
        x_max3d = bounding[nid][2]['x_max']
        y_max3d = bounding[nid][2]['y_max']
        ref_z = np.mean(coor[rows//2, bounding[nid][1]['y_min'].y:bounding[nid][1]['y_max'].y,2])
        length = math.sqrt((y_max3d[0]-y_min3d[0])**2 + (y_max3d[1]-y_min3d[1])**2)
        if max(x_min3d[2], x_max3d[2]) - ref_z > config.block_height and length > config.block_width and abs(x_min3d[2]-x_max3d[2]) > 3:
            full_rate = separate_high_Wall(seg, bounding, nid)
            if full_rate > 0.98 and (max(x_min3d[2], x_max3d[2]) - ref_z < 3.5 or (abs(x_min3d[2]-x_max3d[2])<4.5 and max(x_min3d[2], x_max3d[2]) - ref_z < 2)):
                continue
            seg_building = np.where(seg==nid, building_id, seg_building)
            bounding_building[building_id] = bounding[nid]
            full_rates.append((building_id, full_rate))
            building_id += 1
        else:
            seg_fence = np.where(seg == nid, fence_id, seg_fence)
            bounding_fence[fence_id] = bounding[nid]
            fence_id += 1
    print("    done!")
    with open(os.path.join(save_path, "full_rate.txt"), "w") as f:
        f.write(str(full_rates))

    save_file((seg_fence, bounding_fence), os.path.join(save_path, 'tmp_fence.dat'))
    save_file((seg_building, bounding_building), os.path.join(save_path, 'tmp_building.dat'))
    utils.add_boundbox(save_path, seg_building, bounding_building, flag='building')
    #io.imsave(os.path.join(save_path, 'tmp_building.png'), utils.random_render(seg_building))
    io.imsave(os.path.join(save_path, 'tmp_fence.png'), utils.random_render(seg_fence))
    generate_geojson(path, save_path, bounding_building, name="building")
    print("    done!")
    Count_the_Points(seg_building, save_path)



    return seg_fence, seg_building, bounding_fence, bounding_building


def generate_csv(bounding, scanner="1", flag="mergebefore"):
    print('generate csv... ', end='')
    if os.path.exists("tmp/scanner_{}/segment_line_{}_{}.csv".format(scanner, scanner, flag)):
        os.remove("tmp/scanner_{}/segment_line_{}_{}.csv".format(scanner, scanner, flag))
        
    with open("tmp/scanner_{}/head_info.dat".format(scanner), 'rb') as f:
        data = pickle.load(f)
    o_x = data["original_x"]
    o_y = data["original_y"]
    count=0
    for nid, data in bounding.items():
        _, _, bounding_3d, _ , _= data
        info = "LINESTRING ({} {}, {} {})".format(bounding_3d['y_min'][0]+o_x,bounding_3d['y_min'][1]+o_y, 
                bounding_3d['y_max'][0]+o_x,bounding_3d['y_max'][1]+o_y)
        #info = [bounding_3d['y_min'][0]+o_x,bounding_3d['y_min'][1]+o_y, 
        #        bounding_3d['y_max'][0]+o_x,bounding_3d['y_max'][1]+o_y]
        with open("tmp/scanner_{}/segment_line_{}_{}.csv".format(scanner, scanner, flag),"a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([info])
        count += 1
    print("    done!")


def generate_geojson(path, save_path, bounding, name=None):
    print('generate geojson... ', end='')
    data = load_file('head_info', path)
    coor = load_file('coordinate', path)
    o_x = data["original_x"]
    o_y = data["original_y"]
    o_z = data["original_z"]
    lines = list()
    line_id = list()
    min_h = list()
    max_h = list()
    ref_h = list()
    dis = list()
    depth = list()
    for nid, data in bounding.items():
        _, _, bounding_3d, deep , _= data
        line = LineString([(bounding_3d['y_min'][0]+o_x,bounding_3d['y_min'][1]+o_y),
                           (bounding_3d['y_max'][0]+o_x,bounding_3d['y_max'][1]+o_y)])
        lines.append(line)
        line_id.append(nid)
        depth.append(deep)
        min_h.append(min(bounding_3d['x_min'][2], bounding_3d['x_max'][2])+o_z)
        max_h.append(max(bounding_3d['x_min'][2], bounding_3d['x_max'][2])+o_z)
        dis.append(utils.l2_distance_2d(bounding_3d['y_min'][0:2],bounding_3d['y_max'][0:2]))
        ref_h.append(np.mean(coor[coor.shape[0]//2, bounding[nid][1]['y_min'].y:bounding[nid][1]['y_max'].y,2])+o_z)
    df = pd.DataFrame(line_id,columns=['id'])
    df['min_h'] = min_h
    df['max_h'] = max_h
    df['ref_h'] = ref_h
    df['dis'] = dis
    df['depth'] = depth
    gdf = geopandas.GeoDataFrame(df, geometry=lines)
    if not gdf.empty:
        gdf.to_file(os.path.join(save_path, "segment_line_{}.geojson".format(name)), driver='GeoJSON')
    print("    done!")    
        
    



def run(path, save_path, split_long_seg=False):
    if  not os.path.exists(save_path):
        os.makedirs(save_path)

    depth_image = generate_depth_image(path)

    coor = load_file("coordinate", path)
    reflectance = load_file("reflectance", path)
    reflectance = (reflectance-np.min(reflectance))/float(np.max(reflectance)-np.min(reflectance))
    scene = extract_scene(path)

    building = scene
    region_growing = RegionGrowing(threshold=config.threshold_similarity,
                                   if_4N=config.if_4N, num_filter=config.num_filter)
    seg, avg_norms, bounding = region_growing.run(building, coor, depth_image, reflectance, split_long_seg)
    # bounding (avg_norm, bounding_2d, bounding_3d)
    save_file((seg, bounding), os.path.join(save_path, 'tmp_regiongrowing.dat'))

    seg, bounding = load_file("tmp_regiongrowing", save_path)

    io.imsave(os.path.join(save_path, 'tmp_regiongnotmerge.png'), utils.random_render(seg))
    seg, bounding = region_merge(seg, bounding, path, search_range=config.search_range, split_long_seg=split_long_seg)

    save_file((seg, bounding), os.path.join(save_path, 'tmp_regiongmerge.dat'))
    io.imsave(os.path.join(save_path, 'tmp_regiongmerge.png'), utils.random_render(seg))
    generate_geojson(path, save_path, bounding, name="mergeed")

  
def run_once(path, save_path, file):
    path = os.path.join(path, file)
    save_path = os.path.join(save_path, file)
    run(path, save_path)
    split_building_fence(path, save_path)
    print("done!")















