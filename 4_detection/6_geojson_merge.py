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
import lib.ransac as ransac
import cv2
import lib.find_inliers as find_inliers

root = 'F:/ikg/3_image_generation/all/geojson/'
save_path = 'F:/ikg/3_image_generation/all/'
lines = list()
line_id = list()
min_h = list()
max_h = list()
alphas = list()
geojson_reads=dict()
files = os.listdir(root)
for idx, file in enumerate(files):

   print('{} / {} ...'.format(idx+1, len(files)))
   nid = int(file.split('_')[2].split('.')[0])
   path = os.path.join(root, 'segment_line_{}.geojson'.format(nid))
   with open(path) as f:
      geojson_read = geojson.load(f)
   line = LineString([(geojson_read.features[0].geometry.coordinates[0][0], geojson_read.features[0].geometry.coordinates[0][1]), (geojson_read.features[0].geometry.coordinates[1][0], geojson_read.features[0].geometry.coordinates[1][1])])
   lines.append(line)
   line_id.append(nid)
   min_h.append(geojson_read.features[0].properties['min_h'])
   max_h.append(geojson_read.features[0].properties['max_h'])
   alphas.append(geojson_read.features[0].properties['alpha'])
df = pd.DataFrame(line_id, columns=['id'])
df['min_h'] = min_h
df['max_h'] = max_h
df['alpha'] = alphas
gdf = geopandas.GeoDataFrame(df, geometry=lines)
path_s = os.path.join(save_path, 'geojson_merge')
if not os.path.exists(path_s):
   os.makedirs(path_s)
if not gdf.empty:
   gdf.to_file(os.path.join(path_s, "segment_line.geojson"), driver='GeoJSON')