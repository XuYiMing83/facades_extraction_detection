# -*- coding: utf-8 -*-
import numpy as np
import struct


#####################################################################
# Write binary pointcloud
#####################################################################

dict_pattern = {'double': 'd',
                'float': 'f',
                'short': 'h',
                'int': 'i',
                }

def write_points_bin(points, path, fields, dtypes):
    
    count = len(points)
  
    pattern = ""
    header = """ply\nformat binary_little_endian 1.0\nelement vertex {0}\n""".format(count)
    for f,d in zip(fields, dtypes):
        header += " ".join(["property", d, f, '\n'])
        pattern += dict_pattern[d]
    header += """end_header\n"""
    
    with open(path, "w+") as file_:
        file_.write(header)
        
    with open(path, "a+b") as file_:
        for p in points:
            txt = struct.pack(pattern, *p)
            file_.write(txt)

def write_points_ddd(points, path):
    fields = ["x", "y", "z"]
    dtypes = ['double', 'double', 'double']
    write_points_bin(points, path, fields, dtypes)

def write_points_ddfi(points, path):
    fields = ["x", "y", "z", "run-id"]
    dtypes = ['double', 'double', 'float', 'int']
    write_points_bin(points, path, fields, dtypes)

def write_points_dddf(points, path):
    fields = ["x", "y", "z", "reflectance"]
    dtypes = ['double', 'double', 'double', 'float']
    write_points_bin(points, path, fields, dtypes)

def write_points_dddi(points, path):
    fields = ["x", "y", "z", "seg_id"]
    dtypes = ['double', 'double', 'double', 'int']
    write_points_bin(points, path, fields, dtypes)

def write_points_dddf2(points, path):
    fields = ["x", "y", "z", "reflectance", "seg_id"]
    dtypes = ['double', 'double', 'double', 'float', 'float']
    write_points_bin(points, path, fields, dtypes)

def write_points_dddffff(points, path):
    fields = ["x", "y", "z", "reflectance", "seg_id", "nid", "scalar_Omnivariance_(0.1)",
              "scalar_Surface_variation_(0.1)", "scalar_Normal_change_rate_(0.1)", "label"]
    dtypes = ['double', 'double', 'double', 'float', 'float', 'float', 'float', 'float', 'float', 'float']
    write_points_bin(points, path, fields, dtypes)

def write_points_dddfff(points, path):
    fields = ["x", "y", "z", "reflectance", "seg_id", "nid", "label"]
    dtypes = ['double', 'double', 'double', 'float', 'float', 'float', 'float']
    write_points_bin(points, path, fields, dtypes)

def write_points_dddffR(points, path):
    fields = ["x", "y", "z", "reflectance", "seg_id", "nid", "region"]
    dtypes = ['double', 'double', 'double', 'float', 'float', 'float', 'float']
    write_points_bin(points, path, fields, dtypes)

