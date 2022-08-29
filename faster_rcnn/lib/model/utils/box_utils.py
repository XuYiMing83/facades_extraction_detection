import numpy as np
import torch
from model.rpn.bbox_transform import bbox_overlaps

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



def height_error(preds, gts, iou_thres = 0.5):
    if preds is None or gts is None:
        return None, None

    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(gts, np.ndarray):
        gts = torch.from_numpy(gts)    

    overlap = bbox_overlaps(preds, gts).numpy()

    idx_assigned_gt = overlap.argmax(axis=1)
    confidence = overlap.max(axis=1)
    assigned_gts = gts[idx_assigned_gt]
    flag = np.where(confidence>=iou_thres, 1, 0)
    if np.sum(flag)>0:
        idx_tp = flag.nonzero()[0]
        tps = preds[idx_tp]
        gts = assigned_gts[idx_tp]
    
        preds_edge = tps[:, 3]
        gts_edge = gts[:, 3]
        return preds_edge, gts_edge
    else:
        return None, None

def confusion_matrix(preds, gts, iou_thres = 0.5, self_define=False):
    if preds is None and gts is None:
        return 0, 0, 0
    elif preds is None:
        return 0, 0, len(gts)
    elif gts is None:
        return 0, len(preds), 0
    
    overlap = bbox_overlaps(preds, gts)
    overlap = overlap.numpy()
    idx_assigned_gt = overlap.argmax(axis=1)
    ious = overlap.max(axis=1)
    assigned_gts = gts[idx_assigned_gt]
    flag = np.where(ious>=iou_thres, 1, 0)
    tp = np.count_nonzero(flag)
    fp = len(flag)-tp

    tmp = overlap.max(axis=0)
    fn_flag = np.where(tmp<iou_thres, 1, 0)
    fn = np.count_nonzero(fn_flag)
    fn_inds = fn_flag.nonzero()[0]

    if self_define:
        fp_inds = np.where(ious<iou_thres, 1, 0).nonzero()[0]
        fp_inds = torch.from_numpy(fp_inds).view(-1)
        fp_boxes = preds[fp_inds].numpy().astype(np.uint16)
        fn_inds = torch.from_numpy(fn_inds).view(-1)
        fn_boxes = gts[fn_inds].numpy().astype(np.uint16)
        max_w = int(max(preds[:, 2].max(), gts[:, 2].max()))+1
        max_h = int(max(preds[:, 3].max(), gts[:, 3].max()))+1
        mask_fp = np.zeros((max_h, max_w),dtype=np.int64)
        mask_fn = np.zeros((max_h, max_w),dtype=np.int64)
        id_fp = np.ones((max_h, max_w),dtype=np.int16) * -1
        id_fn = np.ones((max_h, max_w),dtype=np.int16) * -1
        non_fn = []
        non_fp = []
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
                non_fn.append(idx)
                non_fp.extend(nids)
        num_non_fn = len(set(non_fn))
        num_non_fp = len(set(non_fp))
        tp += num_non_fp
        fp -= num_non_fp
        fn -= num_non_fn
    return tp, fp, fn


def customized_box(dets, gts, iou_thres = 0.5):
    """
    input the original detected boxes with scores, return the customized box, see (customized F1)
    det:[x1, y1, x2,y2, scores], gt:[x1, y1, x2, y2]
    """
    if dets is None or gts is None:
        return dets
    preds = dets[:, :-1]
    scores = dets[:, -1]
    max_w = int(max(preds[:, 2].max(), gts[:, 2].max()))+1
    max_h = int(max(preds[:, 3].max(), gts[:, 3].max()))+1
    
    overlap = bbox_overlaps(preds, gts)
    overlap = overlap.numpy()
    idx_assigned_gt = overlap.argmax(axis=1)
    ious = overlap.max(axis=1)    
    tp_inds = np.where(ious>=iou_thres, 1, 0).nonzero()[0]
    tp_inds = torch.from_numpy(tp_inds).view(-1)
    fp_inds = np.where(ious<iou_thres, 1, 0).nonzero()[0]
    fp_inds = torch.from_numpy(fp_inds).view(-1)

    tmp = overlap.max(axis=0)
    fn_inds = np.where(tmp<iou_thres, 1, 0).nonzero()[0]
    fn_inds = torch.from_numpy(fn_inds).view(-1)

    tp_dets = dets[tp_inds].numpy()
    fp_dets = dets[fp_inds].numpy()
    fn_boxes = gts[fn_inds].numpy()

    mask = np.zeros((max_h, max_w),dtype=np.int64)
    id_fp = np.ones((max_h, max_w),dtype=np.int16) * -1
    dict_fp = dict()
    for idx, e in enumerate(fp_dets):
        #mask_fp[e[1]:e[3], e[0]:e[2]] = 1
        id_fp[int(e[1]):int(e[3]), int(e[0]):int(e[2])] = idx
        dict_fp[idx] = e
    output_fp_dets = list()
    for idx, e in enumerate(fn_boxes):
        nids = list(set(id_fp[int(e[1]):int(e[3]), int(e[0]):int(e[2])].reshape(-1)))
        nids = [x for x in nids if x >= 0]
        if len(nids)<=1:
            continue    
        related_dets = list()
        for nid in nids:
            related_dets.append(dict_fp.pop(nid))
        related_dets = np.stack(related_dets)
        outer_det = np.array([related_dets[:, 0].min(), 
                                related_dets[:, 1].min(),
                                related_dets[:, 2].max(), 
                                related_dets[:, 3].max(),
                                related_dets[:, -1].mean()])
        output_fp_dets.append(outer_det)
    output_fp_dets.extend(dict_fp.values())
    if len(output_fp_dets)>0:
        output_fp_dets = np.stack(output_fp_dets)
        res = np.concatenate([tp_dets, output_fp_dets])
        return torch.from_numpy(res)
    else:
        return torch.from_numpy(tp_dets)




