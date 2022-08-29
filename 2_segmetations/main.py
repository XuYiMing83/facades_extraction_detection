# -*- coding: utf-8 -*-

import os

import config
from generate_segment  import run, split_building_fence


root = 'F:/ikg/1_parse/parse_dumpfiles/'  # Path to save header and data files.
if not os.path.exists(root):
    raise InterruptedError("ssd not exist!!")

save_root = 'F:/ikg/6_tmp/tmp_dumpfiles_post/'  # The path to save the data after the segmentation is complete.
if not os.path.exists(save_root):
    os.makedirs(save_root)    
    
files = os.listdir(root)

for idx, file in enumerate(files):

    print("{}:  {} / {}".format(file, idx+1, len(files)))
    path_parse_dumpfile = os.path.join(root, file)
    
    save_path = os.path.join(save_root, file)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    run(path_parse_dumpfile, save_path, split_long_seg=config.threshold_split_long_seg)

    split_building_fence(path_parse_dumpfile, save_path)

