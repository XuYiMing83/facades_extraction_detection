import os
import numpy as np
from functools import reduce
import cv2
from model.rpn.bbox_transform import bbox_overlaps
import torch 
from model.utils.box_utils import point_form, center_size
from model.utils.box_utils import confusion_matrix, customized_box, height_error
from torchvision.ops.boxes import nms


#root = './save/composite_mix/preds_boxes/'
img_root = './data/self_dataset/all_train_image/depth_image/'
preds_root = './save/composite_mix/fpn50/all_train_detection/'
save_root = './save/composite_mix/fpn50/final_train_predictions/' 
fg_mask_root = './data/self_dataset/mask/fg_mask/train/'
occlusion_mask_root = './data/self_dataset/mask/occlusion_mask/train/'

if not os.path.exists(save_root):
    os.makedirs(save_root)

saved_txt = os.path.join(save_root, 'txt')
if not os.path.exists(saved_txt):
    os.makedirs(saved_txt)
saved_image = os.path.join(save_root, 'image')
if not os.path.exists(saved_image):
    os.makedirs(saved_image)

iou_thres = 0.5
custom = False
occlusion_mask=True

def add(x, y):
    return x+0.1*y

def load_filenames(filePath):
    names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'png|txt']
    names = sorted(names, key=lambda x: reduce(add, map(float, x.split('.')[-2].split('_'))), reverse=False)
    paths = [os.path.join(filePath, x) for x in names]
    return paths


img_paths = load_filenames(img_root)
pred_paths = load_filenames(preds_root)

img_names = [a.split('/')[-1].split('.')[0] for a in img_paths]
txt_names = [a.split('/')[-1].split('.')[0] for a in pred_paths]
empty = [a for a in img_names if a not in txt_names]
#assert len(img_paths) == len(txt_paths)

def load_label(path_label):
    res = list()
    with open(path_label, 'r') as file:
        tmp = file.readlines()
        if len(tmp)==0:
            return None, None
        for line in tmp:
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


def draw_according_nid(nid):
    """
    calculating the confusion matrix for single image, if the image is divided 
    into many subimages, merging them at first.  
    """
    path_img = os.path.join(img_root, '{}.png'.format(nid))
    image = cv2.imread(os.path.join(img_root, '{}.png'.format(nid)), 1)
    h, w, c = image.shape
    pred_txts = [x for x in txt_names if int(x.split('_')[0])==nid]
    print(pred_txts)
    pred_txts = sorted(pred_txts, key=lambda x:int(x.split('_')[1]))
    involved_ids = [int((x.split('_')[1])) for x in pred_txts]
    preds = []
    div = round(w/500.0)
    w_ = int(w//max(1, div))
    for n in involved_ids:
        with open(os.path.join(preds_root, '{}_{}.txt'.format(nid, n)), 'r') as f:
            tmp = f.readlines()
        pred = []
        if len(tmp)==0:
            pred = None
        else:
            pred = [np.array(list(map(float, e.split(' '))))[None, :] for e in tmp]
            pred = np.concatenate(pred)
            if n > 0:
              pred[:, 0] = pred[:, 0]+w_*n-100
              pred[:, 2] = pred[:, 2]+w_*n-100
            preds.append(pred)
    if len(preds)>0:
      preds = np.concatenate(preds)

      pred_boxes = preds[:, :-1]
      pred_scores = np.round(preds[:, -1], 2)
      #pred_boxes = preds[:, 0:]
      #pred_scores = np.zeros((pred_boxes.shape[0], ))
    else:
      pred_boxes = None
      pred_scores = None

    pred_boxes = torch.from_numpy(pred_boxes) if pred_boxes is not None else None
    if pred_boxes is not None:
        pred_scores = torch.from_numpy(pred_scores)
        _, order = torch.sort(pred_scores, 0, True)
        cls_dets = torch.cat((pred_boxes, pred_scores[:, None]), 1)
        cls_dets = cls_dets[order]
        keep = nms(pred_boxes[order, :], pred_scores[order], 0.3)
        cls_dets = cls_dets[keep.view(-1).long()]
        if occlusion_mask:
          img_occ_mask = cv2.imread(os.path.join(occlusion_mask_root, '{}.png'.format(nid)), 0).astype(np.int64)
          fg_mask = cv2.imread(os.path.join(fg_mask_root, '{}.png'.format(nid)), 0).astype(np.int64)
          img_occ_mask = np.where(img_occ_mask>0, 255, 0)
          img_occ_mask = np.clip(img_occ_mask-fg_mask, 0, 1).astype(np.int64)
          dets_stack = list()
          for e in cls_dets:
            if np.sum(img_occ_mask[int(e[1]):int(e[3]), int(e[0]):int(e[2])]) < 0.12*(e[3]-e[1])*(e[2]-e[0]):
                dets_stack.append(e)
          cls_dets = torch.stack(dets_stack)

        pred_boxes = cls_dets[:, 0:4]
        pred_scores = cls_dets[:, 4].numpy()
        draw(path_img, saved_image, pred_boxes, pred_scores, nid)
        with open(os.path.join(saved_txt, '{}.txt'.format(nid)), 'w') as f:
          for i, e in enumerate(cls_dets.numpy()):
            f.write(' '.join(map(str, e))+'\n')


def draw(path_img, save_path, preds, pred_scores, nid):
    image = cv2.imread(path_img, 1)
    height, width, _ = image.shape

    if preds is None:
        pass
    else:
        for e in preds:
            cv2.rectangle(image, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), (0, 255, 0)) 

    cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image) 



nids = [int(a.split('/')[-1].split('.')[0]) for a in img_paths]

for i, nid in enumerate(nids):
    #if nid != 708:
    #    continue
    print(i, ' / ', len(nids))
    draw_according_nid(nid)


