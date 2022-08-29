# -*- coding: utf-8 -*-
'''
to get the composite image
'''
import os 
import cv2
import pickle 
import utils
import numpy as np
from skimage import io
'''
syn_images_scale_r: reflectance: utils.rescale
syn_images: reflectance: utils.normalization

'''
root = 'F:/ikg/3_image_generation/{all/'
save_root = 'F:/ikg/3_image_generation/all/composite_image/'


if not os.path.exists(save_root):
    os.makedirs(save_root)

    
geo_path_root = os.path.join(root, 'geometry_image')
depth_path_root = os.path.join(root, 'post_depth_image')
density_path_root = os.path.join(root, 'post_density_image')
reflectance_path_root = os.path.join(root, 'post_reflectance')


def read_file_paths(filePath):
    img_names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'dat|png']
    img_names = sorted(img_names, key=lambda x: int(x.split('.')[-2]), reverse=False)
    img_paths = [os.path.join(filePath, x) for x in img_names]
    return img_paths

def merge_one_image(depth, density, reflectance):
    reflectance = utils.rescale(reflectance)
    reflectance *= 255
    reflectance = reflectance.astype(np.uint8)
    img = cv2.merge([depth, density, reflectance]) 
    return img
    
if __name__ == '__main__':
    path_depth_imgs = read_file_paths(depth_path_root)
    path_density_imgs = read_file_paths(density_path_root)
    path_reflectance_imgs = read_file_paths(reflectance_path_root)
    assert len(path_depth_imgs) == len(path_density_imgs) == len(path_reflectance_imgs)
    for i in range(len(path_depth_imgs)):
        print('\rimage: {}/{}'.format(i, len(path_depth_imgs)), end='')
        path_depth = path_depth_imgs[i]
        path_density = path_density_imgs[i]
        path_reflectance = path_reflectance_imgs[i]
        depth = cv2.imread(path_depth, 0)
        density = cv2.imread(path_density, 0)
        nid = int(path_depth.split('\\')[-1].split('.')[0])
        with open(path_reflectance, 'rb') as f:
            reflectance = pickle.load(f)
        #reflectance = reflectance.astype(np.uint8)
        #io.imsave(os.path.join(save_root_tmp, '{}.png'.format(nid)), density)
        #cv2.imwrite(os.path.join(save_root_ref, '{}.png'.format(nid)), reflectance)   
        syn_img = merge_one_image(depth, density, reflectance)
        cv2.imwrite(os.path.join(save_root, '{}.png'.format(nid)), syn_img)   
    print("    done!")
        