# -*- coding: utf-8 -*-
import os
from dump_process import parse_dump_file


parse_root = 'F:/ikg/1_parse/parse_dumpfiles/'  # Path to save header and data files.

if  not os.path.exists(parse_root):
    os.makedirs(parse_root)    
    

paths = 'E:/'  # The path to save the parse file.
files = os.listdir(paths)

for idx, file in enumerate(files):
    print("{}:  {} / {}".format(file, idx+1, len(files)))
    
    path_dumpfile = os.path.join(paths, file)
    file_name = file.split('.')[0]
    path_parse_file = os.path.join(parse_root, file_name)
    
    if not os.path.exists(path_parse_file):
        os.makedirs(path_parse_file)    
    parse_dump_file(path_dumpfile, path_parse_file)
    
print("done!")


