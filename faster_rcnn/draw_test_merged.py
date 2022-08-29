import os
import numpy as np
from functools import reduce
import cv2
from model.rpn.bbox_transform import bbox_overlaps
import torch 
from model.utils.box_utils import point_form, center_size
from model.utils.box_utils import confusion_matrix, customized_box, height_error
from torchvision.ops.boxes import nms
import pickle

#root = './save/composite_mix/preds_boxes/'
txt_root = './save/composite_mix/fpn50_bp/preds_boxes/epoch_-1/'
img_root = './data/self_dataset/original_image/test/images/'
label_root = './data/self_dataset/original_image/test/labels/'
fg_mask_root = './data/self_dataset/mask/fg_mask/test/'
occlusion_mask_root = './data/self_dataset/mask/occlusion_mask/test/'

height_root = os.path.join(txt_root, 'height_error')
if not os.path.exists(height_root):
    os.makedirs(height_root)

save_root = os.path.join(txt_root, 'images_merged')
if not os.path.exists(save_root):
    os.makedirs(save_root)
save_pred_merged = os.path.join(txt_root, 'prediction_merged')
if not os.path.exists(save_pred_merged):
    os.makedirs(save_pred_merged)

save_pred_custom = os.path.join(txt_root, 'pred_custom_merged')
if not os.path.exists(save_pred_custom):
    os.makedirs(save_pred_custom)

save_tpfpfn = os.path.join(txt_root, 'tpfpfn_merged')
if not os.path.exists(save_tpfpfn):
    os.makedirs(save_tpfpfn)
save_tpfpfn_custom = os.path.join(txt_root, 'tpfpfn_custom')
if not os.path.exists(save_tpfpfn_custom):
    os.makedirs(save_tpfpfn_custom)

save_tpfpfn_occlusion = os.path.join(txt_root, 'tpfpfn_occlusion')
if not os.path.exists(save_tpfpfn_occlusion):
    os.makedirs(save_tpfpfn_occlusion)

iou_thres = 0.5
custom = True # merging inner box
occlusion_mask=False

def add(x, y):
    return x+0.1*y

def load_filenames(filePath):
    names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'png|txt']
    names = sorted(names, key=lambda x: reduce(add, map(float, x.split('.')[-2].split('_'))), reverse=False)
    paths = [os.path.join(filePath, x) for x in names]
    return paths


img_paths = load_filenames(img_root)
txt_paths = load_filenames(txt_root)
label_paths = load_filenames(label_root)

img_names = [a.split('/')[-1].split('.')[0] for a in img_paths]
txt_names = [a.split('/')[-1].split('.')[0] for a in txt_paths]
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


def draw_according_nid(nid, custom=False, occlusion_mask=True):
    """
    calculating the confusion matrix for single image, if the image is divided 
    into many subimages, merging them at first.  
    """
    path_img = os.path.join(img_root, '{}.png'.format(nid))
    image = cv2.imread(os.path.join(img_root, '{}.png'.format(nid)), 1)
    h, w, c = image.shape
    pred_txts = [x for x in txt_names if int(x.split('_')[0])==nid]
    pred_txts = sorted(pred_txts, key=lambda x:int(x.split('_')[1]))
    involved_ids = [int((x.split('_')[1])) for x in pred_txts]
    preds = []
    div = round(w/500.0)
    w_ = int(w//max(1, div))
    for n in involved_ids:
        with open(os.path.join(txt_root, '{}_{}.txt'.format(nid, n)), 'r') as f:
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
    labels, _ = load_label(os.path.join(label_root, '{}.txt'.format(nid)))
    if labels is not None:
      labels = label_convert(labels, w, h)
      labels = point_form(labels)
    
    pred_boxes = torch.from_numpy(pred_boxes) if pred_boxes is not None else None
    labels = torch.from_numpy(labels) if labels is not None else None
    if pred_boxes is not None and labels is not None:
        pred_scores = torch.from_numpy(pred_scores)
        _, order = torch.sort(pred_scores, 0, True)
        cls_dets = torch.cat((pred_boxes, pred_scores[:, None]), 1)
        cls_dets = cls_dets[order]
        keep = nms(pred_boxes[order, :], pred_scores[order], 0.3)
        cls_dets = cls_dets[keep.view(-1).long()]
        with open(os.path.join(save_pred_merged, '{}.txt'.format(nid)), 'w') as f:
          for i, e in enumerate(cls_dets.numpy()):
            f.write(' '.join(map(str, e))+'\n')
        pred_boxes = cls_dets[:, 0:4]
        pred_scores = cls_dets[:, 4].numpy()
        draw(path_img, save_tpfpfn, pred_boxes, labels, pred_scores, nid, False)
        cls_dets = customized_box(cls_dets, labels)
        with open(os.path.join(save_pred_custom, '{}.txt'.format(nid)), 'w') as f:
          for i, e in enumerate(cls_dets.numpy()):
            f.write(' '.join(map(str, e))+'\n')
        if custom:
          img_occ_mask = cv2.imread(os.path.join(occlusion_mask_root, '{}.png'.format(nid)), 0).astype(np.int64)
          fg_mask = cv2.imread(os.path.join(fg_mask_root, '{}.png'.format(nid)), 0).astype(np.int64)
          img_occ_mask = np.where(img_occ_mask>0, 255, 0)
          img_occ_mask = np.clip(img_occ_mask-fg_mask, 0, 1).astype(np.int64)
          if occlusion_mask:
            dets_stack = list()
            for e in cls_dets:
                if np.sum(img_occ_mask[int(e[1]):int(e[3]), int(e[0]):int(e[2])]) < 0.12*(e[3]-e[1])*(e[2]-e[0]):
                    dets_stack.append(e)

            cls_dets = torch.stack(dets_stack)

          pred_boxes = cls_dets[:, 0:4]
          pred_scores = cls_dets[:, 4].numpy()
          draw(path_img, save_tpfpfn_custom if not occlusion_mask else save_tpfpfn_occlusion, 
            pred_boxes, labels, pred_scores, nid, False)
    pred_height, actual_height = height_error(pred_boxes, labels)
    tp, fp, fn = confusion_matrix(pred_boxes, labels, iou_thres, False)
    return tp, fp, fn, pred_height, actual_height



def draw(path_img, save_path, preds, gts, pred_scores, nid, self_defined=False):
    image = cv2.imread(path_img, 1)
    height, width, _ = image.shape

    if preds is None and gts is None:
        pass
    elif preds is None:
        for e in gts:

            cv2.rectangle(image, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), (0, 0, 255)) 
            cv2.putText(image, "fn", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif gts is None:
        for e in preds:
            cv2.rectangle(image, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), (255, 0, 0)) 
            cv2.putText(image, "fp", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:

        for e in gts:
            cv2.rectangle(image, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), (255, 255, 255)) 
            #cv2.putText(image, "gt", (int(max(2, e[2] - 20)), int(max(2, e[1] + 10))),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image) 
        #preds = torch.from_numpy(preds)
        #gts = torch.from_numpy(gts)
        overlap = bbox_overlaps(preds, gts)
        overlap = overlap.numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.numpy()
        if isinstance(gts, torch.Tensor):
            gts = gts.numpy()
        idx_assigned_gt = overlap.argmax(axis=1)
        confidence = overlap.max(axis=1)
        assigned_gts = gts[idx_assigned_gt]
        flag = np.where(confidence>=iou_thres, 1, 0) 
        if np.sum(flag)>0:
            idx_tp = flag.nonzero()[0]
            tp_boxes = preds[idx_tp].astype(np.int16)
            tp_score = pred_scores[idx_tp]
            #cv2.putText(image, str(tp_score[i]), (int(max(2, e[0] + 20)), int(max(2, e[1] + 20))),
            #        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            tp_boxes = None
        flag = np.where(confidence<iou_thres, 1, 0)
        if np.sum(flag)>0:
            idx_fp = flag.nonzero()[0]
            fp_boxes = preds[idx_fp].astype(np.int16)
            fp_score = pred_scores[idx_fp]
        else:
            fp_boxes = None
        tmp = overlap.max(axis=0)
        flag2 = np.where(tmp<iou_thres, 1, 0)
        if np.sum(flag2)>0:
            idx_fn = flag2.nonzero()[0]
            fn_boxes = gts[idx_fn].astype(np.int16)
        else:
            fn_boxes = None    

        non_fn_id = []
        non_fp_id = []
        if self_defined and fn_boxes is not None and fp_boxes is not None:
            max_w = width
            max_h = height
            mask_fp = np.zeros((max_h, max_w),dtype=np.int64)
            mask_fn = np.zeros((max_h, max_w),dtype=np.int64)
            id_fp = np.ones((max_h, max_w),dtype=np.int16) * -1
            id_fn = np.ones((max_h, max_w),dtype=np.int16) * -1
            for idx, e in enumerate(fp_boxes):
                mask_fp[e[1]:e[3], e[0]:e[2]] = 1
                id_fp[e[1]:e[3], e[0]:e[2]] = idx

            for idx, e in enumerate(fn_boxes):
                #mask_fn[e[1]:e[3], e[0]:e[2]] = 1
                fn_area = (e[3]-e[1])*(e[2]-e[0])
                nids = list(set(id_fp[e[1]:e[3], e[0]:e[2]].reshape(-1)))
                nids = [x for x in nids if x >= 0]
                if len(nids)<=1:
                    continue
                fp_area = mask_fp[e[1]:e[3], e[0]:e[2]].sum()
                if float(fp_area)/fn_area > 0.5:
                    non_fn_id.append(idx)
                    non_fp_id.extend(nids)
            add_tp_boxes = fp_boxes[list(set(non_fp_id))]
            if add_tp_boxes is not None and tp_boxes is not None:
                tp_boxes = np.concatenate([tp_boxes, add_tp_boxes], axis=0)
            elif add_tp_boxes is not None:
                tp_boxes = add_tp_boxes
            else:
                pass
        if tp_boxes is not None:
            for i, e in enumerate(tp_boxes):
                cv2.rectangle(image, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), (0, 255, 0)) 
                cv2.putText(image, "tp", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if fp_boxes is not None:    
            for i, e in enumerate(fp_boxes):
                if self_defined:
                    if i in non_fp_id:
                        continue
                cv2.rectangle(image, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), (255, 0, 0)) 
                cv2.putText(image, "fp", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if fn_boxes is not None:    
            for i, e in enumerate(fn_boxes):
                if self_defined:
                    if i in non_fn_id:
                        continue
                cv2.rectangle(image, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), (0, 0, 255)) 
                cv2.putText(image, "fn", (int(max(2, e[0] - 10)), int(max(2, e[1] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        #num_non_fn = len(set(non_fn_id))
        #num_non_fp = len(set(non_fp_id))

    cv2.imwrite(os.path.join(save_path, '{}.png'.format(nid)), image) 



nids = [int(a.split('/')[-1].split('.')[0]) for a in img_paths]
tps = list()
fps = list()
fns = list()
pred_hs = list()
actual_hs = list()
for i, nid in enumerate(nids):
    #if nid != 158:
    #    continue
    print(i, ' / ', len(nids))
    tp, fp, fn , pred_height, actual_height = draw_according_nid(nid, custom=custom, occlusion_mask=occlusion_mask)
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)
    if pred_height is not None:
        pred_hs.append(pred_height)
        actual_hs.append(actual_height)
P = sum(tps)/(sum(tps)+sum(fps)+1e-6)
R = sum(tps)/(sum(tps)+sum(fns)+1e-6)
print(' ')
print("precision:", P)
print('recall:', R)
F1 = (2*P*R)/(P+R+1e-6)
print("F1:", F1)
pred_hs = np.concatenate(pred_hs)#+(config.scann_height-1)
actual_hs = np.concatenate(actual_hs)

error_h = (pred_hs-actual_hs)*0.02
abs_error_h = np.abs(error_h)
print('height error: ', abs_error_h.mean())
if custom and occlusion_mask: 
  with open(os.path.join(height_root, 'error_height_T1T2'), 'wb') as f:
    pickle.dump(error_h, f)
elif custom and not occlusion_mask: 
  with open(os.path.join(height_root, 'error_height_T1'), 'wb') as f:
    pickle.dump(error_h, f)
elif not custom and not occlusion_mask: 
  with open(os.path.join(height_root, 'error_height_none'), 'wb') as f:
    pickle.dump(error_h, f)









