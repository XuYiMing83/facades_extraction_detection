# -*- coding: utf-8 -*-
'''
merge the overlaped facades.  
'''
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
import config

root = 'F:/ikg/6_tmp/'
post_root = os.path.join(root, 'tmp_dumpfiles_post')  # The path to save the data after the segmentation is complete.
save_root = os.path.join(root, 'geojson_split_total_overlap')  # The path to save the data after merge between 2 file.


def centroid(file):
    df = geopandas.read_file(open(file))
    lines1 = list()
    for line in df['geometry']:
        l = list(line.bounds)
        lines1.append(np.array(l)[None,:])
    lines1 = np.concatenate(lines1, axis=0)
    
    b_minx, b_miny = np.min(lines1[:, 0]), np.min(lines1[:, 1])
    b_maxx, b_maxy = np.max(lines1[:, 2]), np.max(lines1[:, 3])
    return (b_minx+b_maxx)/2, (b_miny+b_maxy)/2
    

def if_overlap(df1, df2):
    '''
    whether two blocks have overlap region. 
    '''
    #df1 = geopandas.read_file(open(file1))
    #df2 = geopandas.read_file(open(file2))
    lines1 = list()
    lines2 = list()
    #a = df1.columns.values.tolist()
    for line in df1['geometry']:
        l = list(line.bounds)
        lines1.append(np.array(l)[None,:])
    lines1 = np.concatenate(lines1, axis=0)
    
    b_minx, b_miny = np.min(lines1[:, 0]), np.min(lines1[:, 1])
    b_maxx, b_maxy = np.max(lines1[:, 2]), np.max(lines1[:, 3])
    
    for line in df2['geometry']:
        l = list(line.bounds)
        lines2.append(np.array(l)[None,:])
    lines2 = np.concatenate(lines2, axis=0)
    
    b_minx2, b_miny2 = np.min(lines2[:, 0]), np.min(lines2[:, 1])
    b_maxx2, b_maxy2 = np.max(lines2[:, 2]), np.max(lines2[:, 3])
    if b_minx2 > b_maxx or b_maxx2< b_minx or b_miny2 > b_maxy or b_maxy2 < b_miny:
        return False
    else:
        return True


def merge_df_files(df1, gj2, match):
    df2 = geopandas.read_file(open(gj2))
    if len(df1)==0:
        file_name2 = gj2.split('\\')[-2]
        for i in range(df2.shape[0]):
            if df2.iloc[i]['id'] not in match.keys():
                match[df2.iloc[i]['id']] = [(file_name2, df2.iloc[i]['id'])]
        return df2, match
    file_name2 = gj2.split('\\')[-2]
    repeat = list()
    table2to1 = dict()
    if not if_overlap(df1, df2):
        for j in range(df2.shape[0]):
            if df2.iloc[j]['id']+ df1.iloc[-1]['id'] +1 not in match.keys():
                match[df2.iloc[j]['id']+df1.iloc[-1]['id']+1] = [(file_name2, df2.iloc[j]['id'])]
        df2['id'] += df1.iloc[-1]['id'] + 1
        df1 = df1.append(df2)
        df1 = df1.reset_index(drop=True)
    else:
        matched_id = list()
        base_id = df1.iloc[-1]['id'] + 1
        for i in range(df1.shape[0]):
            for j in range(df2.shape[0]):

                row1 = df1.iloc[i]
                line1 = np.array(row1['geometry']).reshape(-1, )
                row2 = df2.iloc[j]
                line2 = np.array(row2['geometry']).reshape(-1, )

                    #print("leftright:", leftright)
                if abs(utils.slope(np.array(line1[:2]), np.array(line1[2:])) - 
                                   utils.slope(np.array(line2[:2]), np.array(line2[2:]))) > 0.2:
                    continue
                if utils.l2_distance_2d(np.array(line1[:2]), np.array(line2[:2])) > 40:
                    continue
                d_line = utils.l2_distance_lines(np.array(line1[:2]), np.array(line1[2:]), 
                                                 np.array(line2[:2]), np.array(line2[2:]))
                if d_line > 2.5:
                    continue
                ymin_coor,ymax_coor, cover_rate, baseline = \
                    utils.line2line_project(np.array(line1[:2]), np.array(line1[2:]), 
                                            np.array(line2[:2]), np.array(line2[2:]))
                if utils.l2_distance_2d(ymin_coor, ymax_coor) > 30.0:
                    continue
                #if (cover_rate < 1 and cover_rate > 0.7 and d_line < 0.5) or (cover_rate < 0.7 and d_line < 2.5):
                if (cover_rate < config.cover_rate*4 and d_line < 0.5) or (cover_rate < config.cover_rate and d_line < 2.5):
                    if row2['id'] in matched_id:
                        pre_1_id = table2to1[row2['id']]
                        repeat.append(row1['id'])
                        index=df1[df1['id']==pre_1_id].index[0]
                        row_pre = df1.loc[index]
                        line_pre = np.array(row_pre['geometry']).reshape(-1, )
                        ymin_coor,ymax_coor, cover_rate, baseline = \
                            utils.line2line_project(np.array(line_pre[:2]), np.array(line_pre[2:]), 
                                                    np.array(line1[:2]), np.array(line1[2:]))
                        
                        merge_line = LineString([(ymin_coor[0],ymin_coor[1]), (ymax_coor[0],ymax_coor[1])])
                        
                        df1.loc[index,'min_h'] = min(row_pre['min_h'], row1['min_h'])
                        df1.loc[index,'max_h'] = min(row_pre['max_h'], row1['max_h'])
                        df1.loc[index,'ref_h'] = (row1['ref_h']+row_pre['ref_h'])/2
                        df1.loc[index,'depth'] = (row1['depth']+row_pre['depth'])/2
                        df1.loc[index,'dis'] = utils.l2_distance_2d(ymin_coor[:2],ymax_coor[:2])
                        df1.loc[index,'geometry'] = merge_line
                        
                        filter_pair = list()
                        for pair in match[row1['id']]:
                            if pair not in match[row_pre['id']]:
                                filter_pair.append(pair)                        
                        if baseline==1:
                            match[row_pre['id']] = match[row_pre['id']] + filter_pair
                        else:
                            match[row_pre['id']] =  filter_pair + match[row_pre['id']]
                   
                    else:
                        table2to1[row2['id']] = row1['id']
                        matched_id.append(row2['id'])

                        merge_line = LineString([(ymin_coor[0],ymin_coor[1]), (ymax_coor[0],ymax_coor[1])])
                        index=df1[df1['id']==row1['id']].index[0]
                        df1.loc[index,'min_h'] = min(row1['min_h'], row2['min_h'])
                        df1.loc[index,'max_h'] = min(row1['max_h'], row2['max_h'])
                        df1.loc[index,'ref_h'] = (row1['ref_h']+row2['ref_h'])/2
                        df1.loc[index,'depth'] = (row1['depth']+row2['depth'])/2
                        df1.loc[index,'dis'] = utils.l2_distance_2d(ymin_coor[:2],ymax_coor[:2])
                        df1.loc[index,'geometry'] = merge_line
                        
                        if baseline==1:
                            match[row1['id']].append((file_name2, row2['id']))
                        else:
                            match[row1['id']].insert(0, (file_name2, row2['id']))

        
        for i in set(repeat):
            df1.drop(index=df1[df1['id']==i].index[0], inplace=True)
            match.pop(i)
        
        count = 1
        base = df1.iloc[-1]['id']
        for i in range(df2.shape[0]):
            if i in matched_id:
                continue
            row2 = df2.iloc[i]
            tmp_row2_id = row2['id']
            row2['id'] = base + count
            if row2['id'] not in match.keys():
                match[row2['id']] = [(file_name2, tmp_row2_id)]
            df1 = df1.append(row2)
            count += 1
    df1 = df1.reset_index(drop=True)
    return df1, match


def read_all_geojson(root, which='segment_line_building.geojson'):
    file_names = os.listdir(root)
    paths = [os.path.join(root, file) for file in file_names]
    valid_paths = list()
    for path in paths:
        new_path = os.path.join(path, which)
        if os.path.exists(new_path):
            valid_paths.append(new_path)
            
    return valid_paths        
            
        

def nearstneighboor(root_c, remainder):
    min_dis = 10000
    min_center = None
    idx = None
    for i, center in remainder.items():
        dis = utils.l2_distance_2d(root_c, center) 
        if dis < min_dis:
            min_dis = dis
            idx = i
            min_center = center
    return i, min_center
            

def merge(path, which = 'segment_line_building.geojson'):
    
    files = read_all_geojson(path, which)
    file_dict = dict()
    for f in files:
        file_dict[f.split('\\')[-2]] = centroid(f)
        print(file_dict)

    cur_idx = list(file_dict.keys())[0]
    cur_center = file_dict.pop(cur_idx)
    remainder = file_dict
    num_remainder = len(remainder)
    df = pd.DataFrame()
    match = dict()
    r_idx = 1

    while len(remainder) > 0:
        i, n_center = nearstneighboor(cur_center, remainder)

        print("\r{}: {} / {}".format(i, r_idx, num_remainder), end='')
        df, match = merge_df_files(df, os.path.join(path, i, which), match)
        remainder.pop(i)

        cur_center = ((cur_center[0] + n_center[0])/2, (cur_center[1] + n_center[1])/2)
        r_idx += 1

    return df, match



df, match = merge(post_root)
df.to_file(os.path.join(save_root,'merge_v1.geojson'), driver='GeoJSON')

print(df.shape)
print(len(match))
#print(match)


with open(os.path.join(save_root,'merge_v1.txt'),'w') as file_handle:   # .txt可以不自己新建,代码会自动新建
    for k,v in match.items():
        file_handle.write("{}  {}".format(k, v))     # 写入
        file_handle.write('\n')

        
with open(os.path.join(save_root,'merge_v1.dat'),'wb') as f:
    pickle.dump(match, f)
    
print("done!")


