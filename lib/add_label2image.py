# -*- coding: utf-8 -*-
import numpy as np 
import config
import cv2
import os
import torch 


img_root = 'F:/Studienarbeit/ikg_MA-main/xym/ikg/4_detection/all_dataset/train/split_normal/normal_good/images/'
txt_root = 'F:/Studienarbeit/ikg_MA-main/xym/ikg/4_detection/all_dataset/train/split_normal/normal_good/labels/'
save_path = 'F:/Studienarbeit/ikg_MA-main/xym/ikg/4_detection/all_dataset/train/split_normal/labeled_images/'

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted (cx, cy, w, h) form of boxes.
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0

    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] + 0.5 * wh, wh))
    return torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] - 0.5 * boxes[:, 2:],
                                boxes[:, :2] + 0.5 * (boxes[:, 2:] - 2.0)))
    return torch.cat((boxes[:, :2] - 0.5 * boxes[:, 2:],
                      boxes[:, :2] + 0.5 * (boxes[:, 2:] - 2.0)), 1)  # xmax, ymax

def label_convert(labels, width, height):
    """
    from normalized [center_x, center_y, w, h] convert to actual [center_x, center_y, w, h] 
    ImgLabel formal: normalized [center_x, center_y, w, h]
    """
    if labels is None:
        return None
    tmp = np.zeros(labels.shape, dtype=np.int16)
    tmp[:, 0] = np.round(labels[:, 0]*width).astype(np.int16)
    tmp[:, 1] = np.round(labels[:, 1]*height).astype(np.int16)
    tmp[:, 2] = np.round(labels[:, 2]*width).astype(np.int16)
    tmp[:, 3] = np.round(labels[:, 3]*height).astype(np.int16)

    return tmp


def draw(path_img, save_path, boxes, nid):
    image = cv2.imread(path_img, 1)
    if boxes is None:
        cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image) 
        return 
    else:
        for e in boxes:
            e = list(map(int, e))
            cv2.rectangle(image, tuple(e[:2]), tuple(e[2:]), (0, 255, 0))         
    cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image) 
    
    
def read_file_paths(filePath):
    img_names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'txt|png']
    img_names = sorted(img_names, key=lambda x: int(x.split('.')[-2]), reverse=False)
    img_paths = [os.path.join(filePath, x) for x in img_names]
    return img_paths
  
def load_label(path_label):
    res = list()
    with open(path_label, 'r') as file:
        tmp = file.readlines()
        if len(tmp)==0:
            return None
        for line in tmp:
            #if int(line.split(' ')[0])==3:
            #    return None
            a = list(map(float, line.split(' ')[1:]))
            res.append(np.array([a]))
    res = np.concatenate(res, axis=0)
    return res

'''
nid = '3_0'
img_name = '{}.png'.format(nid)
txt_name = '{}.txt'.format(nid)
img_path = os.path.join(img_root, img_name)
txt_path = os.path.join(txt_root, txt_name)
img = cv2.imread(img_path, 0)
h, w = img.shape
labels = load_label(txt_path)
labels = label_convert(labels, w, h)
labels = point_form(labels)
draw(img_path, save_path, labels, nid)
'''