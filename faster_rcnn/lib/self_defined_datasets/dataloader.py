# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from self_defined_datasets.dataset import CityDataSet
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
#from code_DL.lib.fpn.anchor_targets import anchor_target_layer
from model.utils.box_utils import point_form, center_size

from PIL import Image, ImageOps, ImageDraw
from torchvision import transforms


class Blob(object):
    def __init__(self, mode, is_train=True, batch_size=3):
        assert mode in ('train', 'val', 'detection')
        self.mode = mode
        self.is_train = is_train
        self.batch_size = batch_size
        self.imgs = []
        self.gt_boxes = []
        self.classes = []
        self.nids = []
        self.im_sizes = []
        self.train_anchors = []
        self.train_anchor_labels = []
        self.image_unpadded = []
        self.num_boxes = []
        self.img_names = []
        self.img_2 = []

    def append(self, d):
        """
        Adds a single image to the blob
        :param:
        :return:
        """
        i = len(self.imgs)
        if self.mode=='val':
            self.image_unpadded.append(d['image_unpadded'])
        self.imgs.append(d['img'])
        if d['img_2'] is not None:
            self.img_2.append(d['img_2'])
        h, w, scale = d['img_size']
        self.im_sizes.append((h, w, scale))
        if d['boxes'] is not None:
            gt_box = point_form(d['boxes'])
            tmp_gt_box = np.zeros((50, 5), dtype=np.float32)
            num_boxes = min(50, len(gt_box))
            tmp_gt_box[:num_boxes, :4] = gt_box[:num_boxes, :]
            tmp_gt_box[:num_boxes, 4] = 1
            self.gt_boxes.append(tmp_gt_box)
            self.num_boxes.append(num_boxes)
        self.img_names.append(d['img_name'])
        '''
        nid = d['nid'] 
        self.nids.append(nid)
        if d['classes'] is not None:
            self.classes.append(np.column_stack((
            i * np.ones(d['classes'].shape[0], dtype=np.int64),
            d['classes'], 
            nid * np.ones(d['classes'].shape[0], dtype=np.int64))))
        #print('nid:', nid)

        if self.mode in('train', 'val') and d['boxes'] is not None:
            gt_box = point_form(d['boxes'])
            train_anchors_, train_anchor_inds_, train_anchor_targets_, train_anchor_labels_ = \
                anchor_target_layer(gt_box, (h, w))
            
            new_img_PIL = transforms.ToPILImage()(d['img']).convert('RGB')
            draw = ImageDraw.Draw(new_img_PIL)
            line = 3
            for idx, e in enumerate(train_anchors_):
                if train_anchor_labels_[idx]!=1:
                    continue
                x0, y0, x1, y1 = e
                for i in range(1, line + 1):
                    draw.rectangle([x0+(line - i), y0 + (line - i), x1+i, y1+i], outline='green')
            new_img_PIL.show()
            raise InterruptedError()
            self.train_anchors.append(np.hstack((train_anchors_, train_anchor_targets_)))

            self.train_anchor_labels.append(np.column_stack((
                i * np.ones(train_anchor_inds_.shape[0], dtype=np.int64),
                train_anchor_inds_,
                train_anchor_labels_,
            )))     
        '''
    @property        
    def is_exception(self):
        return self.mode in('train', 'val') and (len(self.gt_boxes)==0)

        
    def reduce(self):
        self.imgs = torch.stack(self.imgs, 0)            #[batch_size, 3, IM_SCALE, IM_SCALE]
        if len(self.img_2) > 0:
            self.img_2 = torch.stack(self.img_2, 0)
        else:
            self.img_2 = None
        self.im_sizes = np.stack(self.im_sizes, 0)          # [h ,w, scale]
        self.im_sizes = torch.from_numpy(self.im_sizes)
        if self.mode != 'detection':
            self.gt_boxes = np.stack(self.gt_boxes, 0)  #
            self.gt_boxes = torch.from_numpy(self.gt_boxes).float()
            self.num_boxes = torch.LongTensor(self.num_boxes)

    def to_GPU(self, is_aviable, primary_gpu=0):
        if is_aviable:
            self.imgs = self.imgs.cuda(primary_gpu)
            self.img_2 = self.img_2.cuda(primary_gpu) if self.img_2 is not None else None
            self.im_sizes = self.im_sizes.cuda(primary_gpu)
            if self.mode != 'detection':
                self.gt_boxes = self.gt_boxes.cuda(primary_gpu)
                self.num_boxes = self.num_boxes.cuda(primary_gpu)

    def __call__(self):
        self.to_GPU(torch.cuda.is_available())
        if self.mode == 'train' or self.mode=='val':
            return self.imgs, self.img_2, self.im_sizes, self.gt_boxes, self.num_boxes
        else:
            return self.imgs, self.im_sizes, None, None, None
    
def collate(data, mode, is_train=False):
    blob = Blob(mode, is_train=is_train, batch_size=len(data))
    for d in data:
        blob.append(d)
    blob.reduce()
    #blob.to_GPU(torch.cuda.is_available())
    return blob


class DataLoader(torch.utils.data.DataLoader):

    @classmethod
    def splits(cls, train_data, val_data, test_data=None, batch_size=3, num_workers=1, **kwargs):
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: collate(x, mode='train', is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: collate(x, mode='val', is_train=False),
            drop_last=False,
            # pin_memory=True,
            **kwargs,
        )
        return train_load, val_load
    @classmethod
    def loader(cls, train_data, batch_size=3, num_workers=1, **kwargs):
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: collate(x, mode='detection', is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        return train_load



'''
path_label_train = '/Volumes/Qing Xiao/ikg/4_detection/deep_learning/train/labels/txt_train_label/' 
path_img_train = '/Volumes/Qing Xiao/ikg/4_detection/deep_learning/train/syn_images/' 
path_label_test = '/Volumes/Qing Xiao/ikg/4_detection/deep_learning/val/labels/' 
path_img_test = '/Volumes/Qing Xiao/ikg/4_detection/deep_learning/val/syn_images/'

train_data = CityDataSet('train', path_img_train, path_label_train)
val_data = CityDataSet('val', path_img_test, path_label_test)
print(len(train_data), len(val_data))

train_load, val_load = DataLoader.splits(train_data, val_data, batch_size=2, num_workers=1)
print(len(train_load), len(val_load))
batch = next(iter(train_load))
'''






