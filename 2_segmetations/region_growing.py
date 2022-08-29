# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import numpy as np
import struct
import pickle
from collections import Counter    
from skimage import io
import cv2
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import config

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
  

class RegionGrowing():
    def __init__(self, threshold=0.9, if_4N=True, num_filter=100):
        self.threshold = threshold
        self.if_4N = if_4N
        self.num_filter = num_filter
              
    def selectConnects(self, p):
        """
        @p: if p==1, 24 neighbourhood； p==0, 4 neighbourhood
        """
        if p != 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),\
                        Point(0, 1), Point(-1, 1), Point(-1, 0), Point(-2, -1), Point(-2, 0), Point(-2, 1),Point(2, -1)\
                , Point(2, 0), Point(2, 1), Point(-1, -2), Point(0, -2), Point(1, -2),Point(-1, 2), Point(0, 2),\
                        Point(1, 2), Point(2, 2), Point(-2, 2), Point(2, -2), Point(-2, -2)]
        else:
            connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
        return connects  
      
    def run(self, img, coorinate, depth, reflectance, split_long_seg=False):
        rows, columns = img.shape[0:2]
        search_mask = -np.ones((rows, columns), dtype=np.int16)
        
        seg = -np.ones((rows, columns))
        connects = self.selectConnects(0 if self.if_4N else 1)
            
        seg_id = 0
        
        avg_norms = np.zeros(img.shape)
        boundings = dict()
        
        print("region growing ...")
        for idx_r in tqdm(range(rows)):
            for idx_c in range(columns):
                
                if search_mask[idx_r,idx_c] != -1:
                    continue
                
                if (img[idx_r,idx_c]==np.array([0,0,0])).all():
                    search_mask[idx_r,idx_c] = 1
                    continue
                
                stack = list()
                buffer_list = list()
                stack.append(Point(idx_r,idx_c))
                avg_norm = img[idx_r,idx_c]#np.array([0.0,0.0,0.0])
                avg_depth = 0
                avg_ref = 0.0
                while(len(stack)>0):
                    current_pos = stack.pop()
                    buffer_list.append(current_pos)
                    search_mask[current_pos.x, current_pos.y] = 1
                    for i in range(len(connects)):
                        tmpX = current_pos.x + connects[i].x
                        tmpY = current_pos.y + connects[i].y
                        if search_mask[tmpX, tmpY] != -1:
                            continue
                        if tmpX < 0 or tmpY < 0 or tmpX >= rows-1 or tmpY >= columns-1:
                            continue
                        cur_coorinate = coorinate[current_pos.x, current_pos.y]
                        neighbor_coorinate = coorinate[tmpX, tmpY]

                        #v1 = img[current_pos.x, current_pos.y]
                        v1 = avg_norm / len(buffer_list)
                        v2 = img[tmpX, tmpY]
                        similarity = np.dot(v1,v2)/(np.linalg.norm(v1)*(np.linalg.norm(v2)))
                                           
                        if search_mask[tmpX,tmpY]==-1:
                                                            
                            search_mask[tmpX,tmpY] = 1
                            if similarity > self.threshold and np.sqrt(np.sum((cur_coorinate-neighbor_coorinate)**2))<0.5:
                                
                                stack.append(Point(tmpX,tmpY))
                                avg_norm += img[tmpX,tmpY]
                                avg_depth += depth[tmpX,tmpY]
                                avg_ref += reflectance[tmpX,tmpY]
                search_mask[idx_r,idx_c] = 1
                if len(buffer_list) < self.num_filter:
                   continue
               
                avg_norm = avg_norm/len(buffer_list)
                avg_depth = avg_depth/len(buffer_list)
                avg_ref = avg_ref/len(buffer_list)
                bounding_2d = {'x_max':Point(0,0), 'x_min':Point(rows,0), 
                            'y_max':Point(0,0), 'y_min':Point(0, columns)}
                bounding_3d = dict()
                while(len(buffer_list)>0):
                    cur = buffer_list.pop()
                    seg[cur.x, cur.y] = seg_id
                    avg_norms[cur.x, cur.y] = avg_norm
                    if cur.x > bounding_2d['x_max'].x:
                        bounding_2d['x_max'] = cur
                        bounding_3d['x_max'] = coorinate[cur.x, cur.y]
                    if cur.x < bounding_2d['x_min'].x:
                        bounding_2d['x_min'] = cur
                        bounding_3d['x_min'] = coorinate[cur.x, cur.y]
                    if cur.y > bounding_2d['y_max'].y:
                        bounding_2d['y_max'] = cur
                        bounding_3d['y_max'] = coorinate[cur.x, cur.y]
                    if cur.y < bounding_2d['y_min'].y:
                        bounding_2d['y_min'] = cur
                        bounding_3d['y_min'] = coorinate[cur.x, cur.y]
                if split_long_seg == True:
                # if segment longer than 30m, we should divide it 
                    length = math.sqrt((bounding_3d['y_max'][0]-bounding_3d['y_min'][0])**2 + 
                                       (bounding_3d['y_max'][1]-bounding_3d['y_min'][1])**2)
                    if length > config.threshold_split_long_seg:
                        old_id = seg_id
                        num_part = int(math.ceil(length/20.0))
                        
                        piece_len_2d = int((bounding_2d['y_max'].y - bounding_2d['y_min'].y) // num_part)
                        
                        

                        sub_ymin_3d = bounding_3d['y_min'].copy()
                        for i in range(num_part):
                            sub_bounding_2d = {'x_max':Point(bounding_2d['x_max'].x,bounding_2d['x_max'].y), 
                                               'x_min':Point(bounding_2d['x_min'].x,bounding_2d['x_min'].y), 
                                               'y_max':Point(bounding_2d['y_max'].x, 0), 
                                               'y_min':Point(bounding_2d['y_min'].x, 0)}
                            sub_bounding_3d = {'x_max': bounding_3d['x_max'].copy(), 
                                               'x_min': bounding_3d['x_min'].copy()}
                            
                            sub_ymin_2d = Point(bounding_2d['y_min'].x, bounding_2d['y_min'].y)
                            sub_ymax_2d = Point(bounding_2d['y_max'].x, bounding_2d['y_max'].y)
                            sub_ymax_3d = bounding_3d['y_max'].copy()
                            
                            sub_ymin_2d.y = bounding_2d['y_min'].y + i*piece_len_2d
                             #if i != num_part-1 else bounding_2d['y_max'].y

                            if i != num_part-1:
                                sub_ymax_2d.y = sub_ymin_2d.y + piece_len_2d
                                idx = np.where(seg[sub_bounding_2d['x_min'].x:sub_bounding_2d['x_max'].x, 
                                                   sub_ymax_2d.y]!= -1)[0]
                                
                                if len(idx) == 0:
                                    count = 0
                                    while len(idx) == 0:
                                        count += 1
                                        idx = np.where(seg[sub_bounding_2d['x_min'].x:sub_bounding_2d['x_max'].x, 
                                                           sub_ymax_2d.y+count]!= -1)[0]
                                        
                                    sub_ymax_2d.y += count
                                
                                selected_points = coorinate[sub_bounding_2d['x_min'].x:sub_bounding_2d['x_max'].x, sub_ymax_2d.y]
                                sub_seg = seg[sub_bounding_2d['x_min'].x:sub_bounding_2d['x_max'].x, sub_ymax_2d.y]
                                idx_same_id = np.where(sub_seg==old_id)[0]
                                selected_points  =  selected_points[idx_same_id]
                                if len(selected_points[:, 2]) == 0:
                                    continue
                                sub_ymax_3d = selected_points[np.argmin(selected_points[:, 2])]
                            
                            elif i == num_part-1:
                                sub_ymax_2d.y = bounding_2d['y_max'].y
                                sub_ymax_3d = bounding_3d['y_max'].copy()
                                
                            sub_bounding_2d['y_min'] = sub_ymin_2d
                            sub_bounding_2d['y_max'] = sub_ymax_2d
                            sub_bounding_3d['y_min'] = sub_ymin_3d
                            sub_bounding_3d['y_max'] = sub_ymax_3d

                            cur_id_mask = np.where(seg==old_id, 1, 0)
                            cur_id_mask[:, 0: sub_ymin_2d.y] = 0
                            cur_id_mask[:, sub_ymax_2d.y+1:] = 0
                            seg = np.where(cur_id_mask==1, seg_id, seg)
                            
                            boundings[seg_id] = (avg_norm, sub_bounding_2d.copy(), sub_bounding_3d.copy(), avg_depth, avg_ref)
                            seg_id += 1
                            sub_ymin_3d = sub_ymax_3d.copy()
                            
                    else:
                        boundings[seg_id] = (avg_norm, bounding_2d, bounding_3d, avg_depth, avg_ref)
                        seg_id += 1
                else:
                    boundings[seg_id] = (avg_norm, bounding_2d, bounding_3d, avg_depth, avg_ref)
                    seg_id += 1
            #if flag:
            #    break
        print("region growing done!")        
        return seg, avg_norms, boundings        






