# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from torchvision import transforms

from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from functools import reduce
import pickle

import os
from model.utils.config import cfg

def add(x, y):
    return x+0.1*y

def load_filenames(filePath):
    names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'png|txt']
    names = sorted(names, key=lambda x: reduce(add, map(float, x.split('.')[-2].split('_'))), reverse=False)
    paths = [os.path.join(filePath, x) for x in names]
    return paths

def load_label(path_label):
    res = list()
    with open(path_label, 'r') as file:
        #tmp = file.readlines()
        #if len(tmp)==0:
        #    return None, None
        for line in file.readlines():
            #if int(line.split(' ')[0])==3:
            #    return None
            a = list(map(float, line.split(' ')[1:]))
            res.append(np.array([a]))
    res = np.concatenate(res, axis=0)
    return res, np.array([1]*len(res))


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


class SquarePad(object):
    def __call__(self, img):
        w, h = img.size
        img_padded = ImageOps.expand(img, border=(0, 0, max(h - w, 0), max(w - h, 0)),
                                     fill=(img.getpixel((0,0))))

        return img_padded


class CityDataSet(Dataset):
    def __init__(self, dataset, mode, dual=False, num_train=-1, num_test=-1):

        if mode not in ('val', 'train', 'detection'):
            raise ValueError("Mode must be in test or train. Supplied {}".format(mode))
        self.dataset = dataset
        self.dual = dual
        path_img = './data/self_dataset/{}/{}/images/'.format(self.dataset, mode)
        path_label = './data/self_dataset/{}/{}/labels/'.format(self.dataset, mode)
        #if edge:
        #     path_img = './data/self_dataset/{}/{}/images/'.format(self.dataset, 'val_edge')
        #     path_label = './data/self_dataset/{}/{}/labels/'.format(self.dataset, 'val_edge')
        self.mode = mode
        if self.mode=='detection':
            path_img = './data/self_dataset/all_train_image/split_composite/'

        self.names_img = load_filenames(path_img)

        if self.mode in ('val', 'train'):
            self.names_label = load_filenames(path_label)

            self.filter_empty()
                #self.names_img.sort(key=lambda x: sum(map(float, x.split('/')[-1].split('.')[-2].split('_'))), reverse=False)
                #self.names_label.sort(key=lambda x: sum(map(float, x.split('/')[-1].split('.')[-2].split('_'))), reverse=False)
            assert len(self.names_img) == len(self.names_label)

        tform_composite = [
                SquarePad(),
                Resize(cfg.IM_SCALE),
                ToTensor(),
                Normalize(mean=[0.101935625, 0.016865494, 0.07960085], std=[0.17989501, 0.040014643, 0.1433427]),
                #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet mean and std
            ]
        #elif dataset.startswith('normal'):
        tform_normal = [
                SquarePad(),
                Resize(cfg.IM_SCALE),
                ToTensor(),
                Normalize(mean=[0.50734967, 0.5004253, 0.5000852], std=[0.11684509, 0.12964262, 0.04865679]),  # normal vector
                #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet mean and std
            ]            
        self.transform_composite = Compose(tform_composite)
        self.transform_normal = Compose(tform_normal) 

    def filter_empty(self):

        def is_empty(path_label):
            with open(path_label, 'r') as file:
                tmp = file.readlines()
            return True if len(tmp)==0 else False

        self.names_label = [label for label in self.names_label if not is_empty(label)]
        tmp_saved_idx = [path.split('/')[-1].split('.')[0] for path in self.names_label]
        tmp_imgs = list()
        for path_img in self.names_img:
            if path_img.split('/')[-1].split('.')[0] in tmp_saved_idx:
                tmp_imgs.append(path_img)
        self.names_img = tmp_imgs


        if self.dual:
            tmp_imgs_2 = list()
            for path_img in self.names_img_2:
                if path_img.split('/')[-1].split('.')[0] in tmp_saved_idx:
                    tmp_imgs_2.append(path_img)
            self.names_img_2 = tmp_imgs_2



    def __getitem__(self, index):
        if self.mode=='detection':
            image_unpadded = Image.open(self.names_img[index]).convert('RGB')
            w, h = image_unpadded.size
            img_scale_factor = max(w, h)/cfg.IM_SCALE
            if h > w:
                im_size = (cfg.IM_SCALE, int(w / img_scale_factor), img_scale_factor)
            elif h < w:
                im_size = (int(h / img_scale_factor), cfg.IM_SCALE, img_scale_factor)
            else:
                im_size = (cfg.IM_SCALE, cfg.IM_SCALE, img_scale_factor)
            img = self.transform_composite(image_unpadded)
            entry = {
                'img_size': im_size, 
                'image_unpadded': None,
                'img': img,
                'img_2': None,
                'boxes':  None,
                'classes': None, 
                'fn': self.names_img[index],
                'nid': int(self.names_img[index].split('/')[-1].split('.')[0].split('_')[0]),
                'img_name': self.names_img[index].split('/')[-1].split('.')[0]
                }
            return entry

        s1 = self.names_img[index].split('/')[-1].split('.')[0]
        s2 = self.names_label[index].split('/')[-1].split('.')[0]
        assert s1==s2, 's1: {}, s2: {}'.format(s1, s2)

        image_unpadded = Image.open(self.names_img[index]).convert('RGB')

        w, h = image_unpadded.size
        if self.mode in ('val', 'train'):
            boxes, classes = load_label(self.names_label[index])
            boxes = label_convert(boxes, w, h)

        if self.is_train:
            selection = np.random.uniform(0, 1)
            if selection < 0.33: #scale
                scale = np.random.uniform(0.9, 1.1)
                image_unpadded = image_unpadded.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                if self.dual:
                    image_unpadded_2 = image_unpadded_2.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                boxes = boxes*scale
            elif selection < 0.67 and selection >= 0.33: # flip
                image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
                if self.dual:
                    image_unpadded_2 = image_unpadded_2.transpose(Image.FLIP_LEFT_RIGHT)
                if boxes is not None:
                    boxes[:, 0] = w - boxes[:, 0]
            else:
                pass
            w, h = image_unpadded.size # reclaculate the size 
        
        img_scale_factor = max(w, h)/cfg.IM_SCALE
        if h > w:
            im_size = (cfg.IM_SCALE, int(w / img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h / img_scale_factor), cfg.IM_SCALE, img_scale_factor)
        else:
            im_size = (cfg.IM_SCALE, cfg.IM_SCALE, img_scale_factor)

        #flipped = self.is_train and np.random.random() > 0.5

        if self.mode in ('val', 'train'):
            if boxes is not None:
                boxes = boxes/img_scale_factor
        
        '''
        new_img_PIL = transforms.ToPILImage()(image).convert('RGB')
        draw = ImageDraw.Draw(image_unpadded)
        for e in boxes:
            e = e*600*img_scale
            draw.rectangle([int(e[0]-e[2]//2), int(e[1]-e[3]//2), int(e[0]+e[2]//2), int(e[1]+e[3]//2)],fill=(255, 255, 255))
        image_unpadded.show()
        '''
        if self.dual:
            img = self.transform_composite(image_unpadded)
            img_2 = self.transform_normal(image_unpadded_2)
        else:
            img = self.transform_composite(image_unpadded) if self.dataset.startswith('composite') else self.transform_normal(image_unpadded)
        entry = {
            'img_size': im_size, 
            'image_unpadded': image_unpadded if self.mode=='val' else None,
            'img': img,
            'img_2': img_2 if self.dual else None,
            'boxes': boxes if self.mode in ('val', 'train') else None,
            'classes': classes if self.mode in ('val', 'train') else None, 
            'fn': self.names_img[index],
            'nid': int(self.names_img[index].split('/')[-1].split('.')[0].split('_')[0]),
            'img_name': self.names_img[index].split('/')[-1].split('.')[0]
            }
        return entry
    
    def __len__(self):
        return len(self.names_img)

    @property
    def is_train(self):
        return self.mode.startswith('train')






