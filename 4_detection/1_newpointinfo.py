# -*- coding: utf-8 -*-
'''
to get the composite image
'''
import xlrd
import sys
sys.path.insert(0, '../2_segmetations')
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
import matplotlib.pyplot as plt

from sklearn import linear_model
import random
import lib.ransac as ransac

PATH = "F:/ikg"
parse_root = PATH + '/1_parse/parse_dumpfiles/'  # Path to save header and data files.
root = PATH + '/6_tmp/tmp_dumpfiles_post/'  # The path to save the data after merge_all.
save_root = PATH + '/1_parse/new_parse_dumpfiles/'  # Path to save header and data files with filtered point cloud data.


if  not os.path.exists(save_root):
    os.makedirs(save_root)

files = os.listdir(parse_root)
print(files)
for idx, name in enumerate(files):
    print("{}:  {} / {}".format(name, idx + 1, len(files)))
    with open(os.path.join(root, name, "tmp_building.dat"), 'rb') as f:
        seg, bounding = pickle.load(f)
    with open(os.path.join(parse_root, name, "coordinate.dat"), 'rb') as f:
        coor = pickle.load(f)
    with open(os.path.join(parse_root, name, "reflectance.dat"), 'rb') as f:
        reflectance = pickle.load(f)
    with open(os.path.join(parse_root, name, "normal.dat"), 'rb') as f:
        normal = pickle.load(f)
    with open(os.path.join(parse_root, name, "head.dat"), 'rb') as f:
        head = pickle.load(f)
    with open(os.path.join(parse_root, name, "head_info.dat"), 'rb') as f:
        header = pickle.load(f)

    file_name = name.split('.')[0]
    path_parse_file = os.path.join(save_root, file_name)

    if not os.path.exists(path_parse_file):
        os.makedirs(path_parse_file)

    seg2 = np.expand_dims(seg, 2).repeat(3, axis=2)
    new_coor = np.where(seg2 >= 0, coor, 0)
    new_reflectance = np.where(seg >= 0, reflectance, 0)
    new_normal = np.where(seg2 >= 0, normal, 0)
    new_head = np.where(seg2 >= 0, head, 0)

    with open(os.path.join(save_root, name, "coordinate.dat"), "wb") as f:
        pickle.dump(new_coor, f)
    with open(os.path.join(save_root, name, "reflectance.dat"), "wb") as f:
        pickle.dump(new_reflectance, f)
    with open(os.path.join(save_root, name, "normal.dat"), "wb") as f:
        pickle.dump(new_normal, f)
    with open(os.path.join(save_root, name, "head.dat"), "wb") as f:
        pickle.dump(new_head, f)
    with open(os.path.join(save_root, name, "head_info.dat"), "wb") as f:
        pickle.dump(header, f)
