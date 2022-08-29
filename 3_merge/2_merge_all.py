# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../all_segmetations')
import pickle
import geopandas
import os
import numpy as np
import math
import config
from skimage import io
import utils
from shapely.geometry import LineString

root = 'F:/ikg/6_tmp/geojson_split_total_overlap/'  # The path to save the data after merge between 2 file. And save merge_all
def mode(arr):
    sum_error = list()
    for i in range(len(arr)):
        sum_error.append(sum([abs(arr[i]-x) for x in arr]))
    idx = sum_error.index(min(sum_error))
    return idx
       

with open(os.path.join(root, 'merge_v1.dat'), 'rb') as f:
    match = pickle.load(f)

df = geopandas.read_file(open(os.path.join(root, 'merge_v1.geojson')))

points = list()
points_dict = dict()
slop = dict()
distance = dict()
popped_id = list()

for i in range(df.shape[0]):
    row = df.iloc[i]
    line = np.array(row['geometry']).reshape(-1, )
    center = [(line[2] + line[0]) /2.0, (line[3] + line[1]) /2.0]
    point = (center, row['id'])
    slop[row['id']] = (line[3] + line[1])/(line[2] + line[0])
    distance[row['id']] = row['dis']
    points.append(point)
    points_dict[row['id']] = point

points = sorted(points, key=lambda x:x[0][0])  
  
while len(points) > 0:
    print('\r...{}'.format(len(points)))
    cur_coor, cur_id = points.pop(0)
    if cur_id in popped_id:
        continue
    #print("in cur:", cur_id)
    cur_neighboor = [(cur_coor, cur_id)]

    #line1 = np.array(row['geometry']).reshape(-1, )
    cur_slops = [slop[cur_id]]
    cur_dis = [distance[cur_id]]
    for coor, nid in points:
        if utils.l2_distance_2d(coor, cur_coor) < 10:
            cur_neighboor.append((coor, nid))
            cur_slops.append(slop[nid])
            cur_dis.append(distance[nid])
    
    if len(cur_neighboor) == 1:
        continue    

    best_index = mode(cur_slops)
    #print("best_index:", best_index)
    try:
        index_1 = df[df['id']==cur_neighboor[best_index][1]].index[0]
    except:
        print(cur_neighboor[best_index][1])
        #print(df[df['id']==cur_neighboor[best_index][1]])
        raise InterruptedError()
    row1 = df.loc[index_1]
    #print('all id:', [x[1] for x in cur_neighboor])
    for idx in range(len(cur_neighboor)):
        
        if idx == best_index:
            continue
        point = cur_neighboor[idx]
        
        line1 = np.array(row1['geometry']).reshape(-1, )
        row2 = df.loc[df[df['id']==point[1]].index[0]]
        line2 = np.array(row2['geometry']).reshape(-1, )   
        
    
            #line2 = np.array(df.iloc[df['id']==point[1]]['geometry']).reshape(-1, )
        if abs(slop[row2['id']] - slop[row1['id']]) < 0.2:
                d_line = utils.l2_distance_lines(line1[:2], line1[2:], line2[:2], line2[2:])
                if d_line < 2.5:
                    ymin_coor,ymax_coor, cover_rate, baseline = \
                    utils.line2line_project(np.array(line1[:2]), np.array(line1[2:]), 
                                            np.array(line2[:2]), np.array(line2[2:]))
                    if utils.l2_distance_2d(ymin_coor, ymax_coor) > 30.0:
                        continue
                    #if cover_rate < config.cover_rate:
                    if (cover_rate < config.cover_rate*4 and d_line < 0.5) or (cover_rate < config.cover_rate and d_line < 2.5):
    
                        try:
                            points.remove(point)
                            #print("in cover_rate:", point[1])
                        except:
                            pass
                        
                        merge_line = LineString([(ymin_coor[0],ymin_coor[1]), (ymax_coor[0],ymax_coor[1])])
                        if baseline==1:
                            index=df[df['id']==row1['id']].index[0]
                        else:
                            index=df[df['id']==row2['id']].index[0]
                        df.loc[index,'min_h'] = min(row1['min_h'], row2['min_h'])
                        df.loc[index,'max_h'] = min(row1['max_h'], row2['max_h'])
                        df.loc[index,'ref_h'] = (row1['ref_h']+row2['ref_h'])/2
                        df.loc[index,'depth'] = (row1['depth']+row2['depth'])/2
                        df.loc[index,'dis'] = utils.l2_distance_2d(ymin_coor[:2],ymax_coor[:2])
                        df.loc[index,'geometry'] = merge_line
                        #print('xq:', row1['id'], row2['id'], baseline)
                        if baseline==1:
                            match[row1['id']] = match[row1['id']] + match[row2['id']]
                            df = df.drop(df[df['id']==row2['id']].index)
                            popped_id.append(row2['id'])
                            try:
                                points.remove(points_dict[row2['id']])
                            except:
                                pass
                            
                            match.pop(row2['id'])
                        else:
                            match[row2['id']] = match[row2['id']] + match[row1['id']]
                            df = df.drop(df[df['id']==row1['id']].index)
                            popped_id.append(row1['id'])
                            try:
                                points.remove(points_dict[row1['id']])
                            except:
                                pass                           
                            match.pop(row1['id'])
                            row1 = df.loc[df[df['id']==row2['id']].index[0]]
    

print("    done!")



df.to_file(os.path.join(root, 'merge_post.geojson'), driver='GeoJSON')

print(df.shape)
print(len(match))
#print(match)
with open(os.path.join(root, 'merge_post.txt'),'w') as file_handle:   # .txt可以不自己新建,代码会自动新建
    for k,v in match.items():
        file_handle.write("{}  {}".format(k, v))     # 写入
        file_handle.write('\n')

        
with open(os.path.join(root, 'merge_post.dat'),'wb') as f:
    pickle.dump(match, f)
    
print("done!")
