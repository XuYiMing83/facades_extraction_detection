# -*- coding: utf-8 -*-

# segmentation
normal_vector_facade = 0.95  # filter points whose normals are not perpendicular to the ground
modality_reference_average = 'mode'  # median , mode, mean
threshold_split_long_seg = 20
# for region growing
threshold_similarity = 0.95 # the similarity between two norms.
if_4N = False  # using 4 neighboors or 8 neighboors for region growing
num_filter = 30 # if the pixel of region less than this threshold, then remove it.

# for region merging 
search_range = 20 
using_depth = False # if true, then using depth as threshold for merging, otherwise using space distance
threshold_dis = 1.5
inliers_thres = 0.8
max_iterations = 50
iou_thres = 0.5
scann_height = 5
sum_nid_points = 1300

# for split fence and building 
block_height = 3.0
block_width = 3.0
split_height = 3.0
threshold_height = 5.0  

image_overlap = 100 # for split the long image


SCALE = 50


cover_rate = 0.1
