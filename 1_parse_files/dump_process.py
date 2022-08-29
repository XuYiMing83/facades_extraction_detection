# -*- coding: utf-8 -*-

import numpy as np
import struct
import pickle
import os
from tqdm import tqdm

def parse_dump_file(file_path, parse_path):
    
    with open(file_path, "rb") as file:
        data = file.read()
        a = data[0:8]
        rows, columns = struct.unpack("2I", a)
        
        a = data[8:32]
        origin_x,origin_y,origin_z = struct.unpack("3d", a)
        with open(os.path.join(parse_path, 'head_info.dat'), "wb") as f:
            pickle.dump({"rows":rows, "columns":columns, 'original_x':origin_x,
                         'original_y':origin_y,'original_z':origin_z}, f)
        
        reflectance = np.zeros((columns*rows,), dtype=np.int16)
        norms = np.zeros((columns*rows, 3),dtype=np.float32)
        coor = np.zeros((columns*rows, 3),dtype=np.float32)  # Point coordinates
        head = np.zeros((columns*rows, 3),dtype=np.float32)  # Head coordinates
        d_ = np.zeros((columns*rows,),dtype=np.float32)
        for i in tqdm(range(rows*columns)):
                #pbar.set_description("Processing {}".format(file_path.split('/')[-1]))
                record = data[32+i*52:32+(1+i)*52]
                de_record = struct.unpack("3f3f1d4fhH", record)
                #print(de_record)
                #height[i] = de_record[2]
                reflectance[i] = de_record[-2]
                
                norms[i] = np.array(de_record[7:10])
                d_[i] = de_record[10]
                coor[i] = np.array(de_record[0:3])
                head[i] = np.array(de_record[3:6])
   
        norms = np.reshape(norms, (columns, rows, 3)).transpose(1,0,2)
        coor = np.reshape(coor, (columns, rows, 3)).transpose(1,0,2)
        head = np.reshape(head, (columns, rows, 3)).transpose(1,0,2)
        reflectance = np.reshape(reflectance, (columns, rows)).transpose(1,0)
        d_ = np.reshape(d_, (columns, rows)).transpose(1,0)
        
        with open(os.path.join(parse_path, 'normal.dat'), "wb") as f:
            pickle.dump(norms, f)
        with open(os.path.join(parse_path, 'reflectance.dat'), "wb") as f:
            pickle.dump(reflectance, f)

        with open(os.path.join(parse_path, 'head.dat'), "wb") as f:
            pickle.dump(head, f)
            
        with open(os.path.join(parse_path, 'coordinate.dat'), "wb") as f:
            pickle.dump(coor, f)
        with open(os.path.join(parse_path, 'd_.dat'), "wb") as f:
            pickle.dump(d_, f)
                
        
#parse_dump_file("../1/Data_MA/dumpfiles/190906_074826_Scanner_1.dump")
#parse_dump_file("../1/Data_MA/dumpfiles/190906_074826_Scanner_2.dump")









