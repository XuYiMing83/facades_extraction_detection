# -*- coding: utf-8 -*-
import numpy as np
#from lib.box_intersections_cpu.bbox import bbox_overlaps as bbox_overlaps_np
#from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import math
import pickle
import warnings
import os
import config
import matplotlib.pyplot as plt


import geopandas
from shapely import geometry
from shapely.geometry import LineString
import pandas as pd
import pcl
#import point_cloud_utils as pcu


def random_render(seg):
    warnings.filterwarnings('ignore')
    rows, columns = seg.shape
    set_id = set(seg.reshape(-1))
    color_dict = {nid:np.random.randint(0, 255, (3,)) for nid in set_id}
    render = np.zeros((rows, columns, 3))
    for i in range(rows):
        print("\rrandom render: {:.2f}%".format(100 * i / float(rows)), end='')
        for j in range(columns):
            if seg[i,j] == -1:
                continue
            #print(color_dict[seg[i,j]])
            render[i,j] = color_dict[seg[i,j]]
    print("    done!")
    return render


def add_boundbox(path, seg, bounding, flag="building", color="red"):
    render_seg=random_render(seg)
    render_seg = render_seg*255
    img = Image.fromarray(render_seg.astype('uint8')).convert('RGB')
    draw = ImageDraw.Draw(img)
    for idx, (nid, data) in enumerate(bounding.items()):
        print("\radd bounding box: {:.2f}%".format(100 * idx / float(len(bounding))), end='')
        norm, bounding2d, bounding3d, _ , _= data
        draw_box(draw, bounding2d['x_min'].x, bounding2d['x_max'].x,
                 bounding2d['y_min'].y, bounding2d['y_max'].y, color)
        draw.text((bounding2d['y_min'].y,bounding2d['x_min'].x), "{}".format(nid),  
                  fill=(255,255,255))
    #draw.set_color(img,[rr,cc],[255,0,0])
    img.save(os.path.join(path ,"bounding_box_{}.png".format(flag)))
    print("    done!")

def draw_box(draw, x_min, x_max, y_min, y_max, color='red'):
    draw.line((y_min, x_min, y_max, x_min),color)
    draw.line((y_max, x_min, y_max, x_max),color)
    draw.line((y_max, x_max, y_min, x_max),color)
    draw.line((y_min, x_max, y_min, x_min),color)


def slope(p1, p2):
    return (p1[1]-p2[1])/(p1[0]-p2[0])    


def l2_distance_2d(a, b):
    return math.sqrt((a[1] - b[1])**2 + (a[0] - b[0])**2)


def l2_distance_lines(p11,p12, p21, p22):
    """
    p11: A['y_min'], p12: A['y_max'] 
    p21: B['y_min'], p22: B['y_max'] 
    """
    A_n = p12[1] - p11[1]
    B_n = -p12[0] + p11[0]
    C_n = p12[0]*p11[1] - p11[0]*p12[1]
            
    x0_c = p21[0]
    y0_c = p21[1]
    d1 = abs(A_n*x0_c+B_n*y0_c+C_n)/math.sqrt(A_n**2+B_n**2)
    x0_c_1 = p22[0]
    y0_c_1 = p22[1]
    d2 = abs(A_n*x0_c_1+B_n*y0_c_1+C_n)/math.sqrt(A_n**2+B_n**2)
    return max(d1, d2)


def l2_distance_4_points(p11,p12, p21, p22):
    """
    p11: A['y_min'], p12: A['y_max'] 
    p21: B['y_min'], p22: B['y_max'] 
    """
    a = math.sqrt((p11[1] - p21[1])**2 + (p11[0] - p21[0])**2)
    b = math.sqrt((p11[1] - p22[1])**2 + (p11[0] - p22[0])**2)
    c = math.sqrt((p12[1] - p21[1])**2 + (p12[0] - p21[0])**2)
    d = math.sqrt((p12[1] - p22[1])**2 + (p12[0] - p22[0])**2)
    return min(min(a, b), min(c, d))



def line2line_project(p11, p12, p21, p22):
    """
    return the new line after projection
    """
    l1 = l2_distance_2d(p11, p12)
    l2 = l2_distance_2d(p21, p22)
    if np.dot(p21-p22, p11-p12) < 0:
        tmp = p21
        p21 = p22
        p22 = tmp
    if l1 <= l2:  # p11, p12 project to line2
        k = (p22[1]-p21[1])/(p22[0]-p21[0])
        b = -k*p21[0] + p21[1]
        x1 = (k*(p11[1]-b) + p11[0])/(k**2+1)
        y1 = k*x1 + b
        x2 = (k*(p12[1]-b) + p12[0])/(k**2+1)
        y2 = k*x2 + b
        min_coor = np.array([x1, y1]) if np.dot(p21-p22, p21-np.array([x1, y1])) < 0 else p21
        max_coor = np.array([x2, y2]) if np.dot(p22-p21, p22-np.array([x2, y2])) < 0 else p22
        baseline = 2
    else:
        k = (p12[1]-p11[1])/(p12[0]-p11[0])   
        b = -k*p11[0] + p11[1]
        x1 = (k*(p21[1]-b) + p21[0])/(k**2+1)
        y1 = k*x1 + b
        x2 = (k*(p22[1]-b) + p22[0])/(k**2+1)
        y2 = k*x2 + b
        min_coor = np.array([x1, y1]) if np.dot(p11-p12, p11-np.array([x1, y1])) < 0 else p11
        max_coor = np.array([x2, y2]) if np.dot(p12-p11, p12-np.array([x2, y2])) < 0 else p12
        baseline = 1
    if min(l1, l2) != 0:
        cover_rate = (l2_distance_2d(min_coor, max_coor) - max(l1, l2))/min(l1, l2)
    else:
        cover_rate = 99999
    min_coor_bp = np.zeros((3, ))
    min_coor_bp[:2] = min_coor
    max_coor_bp = np.zeros((3,))
    max_coor_bp[:2] = max_coor
    return min_coor_bp, max_coor_bp, cover_rate, baseline



def generate_depth_image(scanner="1"):
    with open("tmp/scanner_{}/coordinate.dat".format(scanner), "rb") as f:
            coor = pickle.load(f)
    with open("tmp/scanner_{}/head.dat".format(scanner), "rb") as f:
            head = pickle.load(f)
    tmp = coor - head
    tmp = tmp**2
    tmp = tmp.sum(axis=2)
    dis = np.sqrt(tmp)
    return dis


def rescale(data):
    tmp = (data - np.min(data))/(np.max(data) - np.min(data))
    return tmp

def standardization(data):
    tmp = (data - np.mean(data)) / np.std(data,ddof=1)
    tmp = tmp - np.min(tmp)
    return tmp

def label_convert(labels, width, height):  
    # normalized [cx, cy, w, h] to [x1, y1, x2,y2] 
    if labels is None:
        return None
    '''
    tmp = np.zeros(labels.shape, dtype=np.int16)
    convert = np.zeros(labels.shape, dtype=np.int16)
    tmp[:, 0] = np.round(labels[:, 0]*width).astype(np.int16)
    tmp[:, 1] = np.round(labels[:, 1]*height).astype(np.int16)
    tmp[:, 2] = np.round(labels[:, 2]*width/2).astype(np.int16)
    tmp[:, 3] = np.round(labels[:, 3]*height/2).astype(np.int16)
    convert[:, 0] = tmp[:, 0] - tmp[:, 2]
    convert[:, 1] = tmp[:, 1] - tmp[:, 3]
    convert[:, 2] = tmp[:, 0] + tmp[:, 2]
    convert[:, 3] = tmp[:, 1] + tmp[:, 3]
    '''
    tmp = np.zeros(labels.shape, dtype=np.float64)
    convert = np.zeros(labels.shape, dtype=np.float64)
    tmp[:, 0] = labels[:, 0]*width
    tmp[:, 1] = labels[:, 1]*height
    tmp[:, 2] = labels[:, 2]*width/2
    tmp[:, 3] = labels[:, 3]*height/2
    convert[:, 0] = tmp[:, 0] - tmp[:, 2]
    convert[:, 1] = tmp[:, 1] - tmp[:, 3]
    convert[:, 2] = tmp[:, 0] + tmp[:, 2]
    convert[:, 3] = tmp[:, 1] + tmp[:, 3]
    return convert    

def label_convert_coco(labels, width, height):  
    #[x1, y1, x2,y2] to normalized [cx, cy, w, h]
    if labels is None:
        return None
    tmp = np.zeros(labels.shape, dtype=np.float32)
    tmp[:, 2] = (labels[:, 2] - labels[:, 0])/float(width)
    tmp[:, 3] = (labels[:, 3] - labels[:, 1])/float(height)
    tmp[:, 0] = (labels[:, 2] + labels[:, 0])/(2*float(width))
    tmp[:, 1] = (labels[:, 3] + labels[:, 1])/(2*float(height))   
    return tmp
 


def bbox_overlaps(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    [x1, y1, x2, y2]
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        #return bbox_overlaps_np(box_a, box_b)

def confusion_matrix(preds, gts):
    if preds is None and gts is None:
        return 0, 0, 0
    elif preds is None:
        return 0, 0, len(gts)
    elif gts is None:
        return 0, len(preds), 0
    
    overlap = bbox_overlaps(preds, gts)
    idx_assigned_gt = overlap.argmax(axis=1)
    confidence = overlap.max(axis=1)
    assigned_gts = gts[idx_assigned_gt]
    flag = np.where(confidence>=config.iou_thres, 1, 0)
    tp = np.count_nonzero(flag)
    fp = len(flag)-tp

    tmp = overlap.max(axis=0)
    flag2 = np.where(tmp<config.iou_thres, 1, 0)
    fn = np.count_nonzero(flag2)
    return tp, fp, fn

def prediction_filter(preds, cand, iou=0.3):
    if cand is None:
        return None
    elif preds is None:
        return cand
    overlap = bbox_overlaps(cand, preds)
    idx_assigned_pred = overlap.argmax(axis=1)
    confidence = overlap.max(axis=1)
    assigned_cand = preds[idx_assigned_pred]   
    idx = np.where(confidence>=iou)[0]
    res = assigned_cand[idx]
    return res

def height_error(preds, gts):
    if preds is None and gts is None:
        return None, None
    elif preds is None:
        return None, None
    elif gts is None:
        return None, None
    
    overlap = bbox_overlaps(preds, gts)
    idx_assigned_gt = overlap.argmax(axis=1)
    confidence = overlap.max(axis=1)
    assigned_gts = gts[idx_assigned_gt]
    flag = np.where(confidence>=config.iou_thres, 1, 0)
    if np.sum(flag)>0:
        idx_tp = flag.nonzero()[0]
        tps = preds[idx_tp]
        gts = assigned_gts[idx_tp]
    
        preds_edge = tps[:, 3]
        gts_edge = gts[:, 3]
        return preds_edge, gts_edge
    else:
        return None, None

def window_line_cp(preds, gts, averages, info_path, save_path, nid):
    """
    generate cloud points for detected window line, in local system instead in global 
    """
    if preds is None or gts is None:
        return 
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    #ref_plane = info['ref_plane']
    buttom_edge = info['buttom_edge']
    left_edge = info['left_edge']
    Trans = info['trans_i2o']
    rows, cols = averages.shape
    overlap = bbox_overlaps(preds, gts)
    idx_assigned_gt = overlap.argmax(axis=1)
    confidence = overlap.max(axis=1)
    assigned_gts = gts[idx_assigned_gt]
    flag = np.where(confidence>=config.iou_thres, 1, 0)
    if np.sum(flag)>0:
        idx_tp = flag.nonzero()[0]
        tps = preds[idx_tp]
    else:
        tps = []
    flag = np.where(confidence<config.iou_thres, 1, 0)
    if np.sum(flag)>0:
        idx_fp = flag.nonzero()[0]
        fps = preds[idx_fp]    
    else:
        fps = []
    tp_points = list()
    for e in tps:
        x0, y0, x1, y1 = e
        x = np.arange(x0, x1, 0.01, dtype=np.float32)*0.02 + left_edge
        z = (np.ones(x.shape, dtype=np.float32)*(rows-y1))*0.02+buttom_edge
        y = (np.ones(x.shape, dtype=np.float32)*averages[(y1+y0)//2, (x0+x1)//2])
        points = np.zeros((len(x), 3), dtype=np.float32)
        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z
        p2 = points.copy()
        p2[:, 2] += 0.01 
        p3 = points.copy()
        p3[:, 1] += 0.01 
        #points = np.concatenate([x, y, z], axis=1)
        tp_points.append(points)
        tp_points.append(p2)
        tp_points.append(p3)
    if len(tp_points)==0:
        tp_points = None
    else:
        tp_points = np.concatenate(tp_points, axis=0)
    
    fp_points = list()
    for e in fps:
        x0, y0, x1, y1 = e
        x = np.arange(x0, x1, 0.01, dtype=np.float32)*0.02 + left_edge
        z = (np.ones(x.shape, dtype=np.float32)*(rows-y1))*0.02+buttom_edge
        y = (np.ones(x.shape, dtype=np.float32)*averages[(y1+y0)//2, (x0+x1)//2])
        points = np.zeros((len(x), 3), dtype=np.float32)
        points[:, 0] = x.copy()
        points[:, 1] = y.copy()
        points[:, 2] = z.copy()
        #points = np.concatenate([x, y, z], axis=1)
        p2 = points.copy()
        p2[:, 2] += 0.01 
        p3 = points.copy()
        p3[:, 1] += 0.01 
        #points = np.concatenate([x, y, z], axis=1)
        fp_points.append(points)
        fp_points.append(p2)
        fp_points.append(p3)
        
    if len(fp_points)==0:
        fp_points = None
    else:
        fp_points = np.concatenate(fp_points, axis=0)
    
    basex = info['original_x']
    basey = info['original_y']
    basez = info['original_z']   

    cloud = pcl.PointCloud()    
    if tp_points is not None:
        tp_points_aug = np.ones((len(tp_points), 4), dtype=np.float32)
        tp_points_aug[:, :3] = tp_points
        o_tp_points = Trans @ tp_points_aug.T 
        o_tp_points = o_tp_points.T[:, 0:3]
        '''
        o_tp_points[:, 0] += basex
        o_tp_points[:, 1] += basey
        o_tp_points[:, 2] += basez
        '''
        cloud.from_array(o_tp_points)
        pcl.save(cloud, os.path.join(save_path, '{}_tp.ply'.format(nid)), format="ply")
    
    if fp_points is not None:
        fp_points_aug = np.ones((len(fp_points), 4), dtype=np.float32)
        fp_points_aug[:, :3] = fp_points
        o_fp_points = Trans @ fp_points_aug.T 
        o_fp_points = o_fp_points.T[:, 0:3]
        '''
        o_fp_points[:, 0] += basex
        o_fp_points[:, 1] += basey
        o_fp_points[:, 2] += basez
        '''
        cloud.from_array(o_fp_points)
        pcl.save(cloud, os.path.join(save_path, '{}_fp.ply'.format(nid)), format="ply")
    
    
    

def window_line(preds, depths, averages, info_path, nid):
    if preds is None:
        return pd.DataFrame()
    
    rows, cols = depths.shape
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    #ref_plane = info['ref_plane']
    buttom_edge = info['buttom_edge']
    left_edge = info['left_edge']
    Trans = info['trans_i2o']
    #print(left_edge, buttom_edge)
    X0 = 1/config.SCALE*preds[:, 0]+left_edge
    X1 = 1/config.SCALE*preds[:, 2]+left_edge
    Z = 1/config.SCALE*(rows-preds[:, 3]) + buttom_edge

    center = list(zip((preds[:, 0]+preds[:, 2])//2, preds[:, 3]))

    Y= list()
    for i in center:
        Y.append(averages[int(min(i[1], rows-1)), int(min(i[0], cols-1))])
    Y = np.array(Y)
    start = np.zeros((len(X0), 4))
    end = np.zeros((len(X0), 4))
    start[:, 3] = 1
    end[:, 3] = 1
    start[:, 2] = Z.copy()
    end[:, 2] = Z.copy()
    start[:, 0] = X0
    start[:, 1] = Y.copy()
    end[:, 0] = X1
    end[:, 1] = Y.copy()
    
    o_start = Trans @ start.T 
    o_start = o_start.T[:, 0:3]
    o_end = Trans @ end.T
    o_end = o_end.T[:, 0:3]
    basex = info['original_x']
    basey = info['original_y']
    basez = info['original_z']
    lines = list()
    widths = X1 - X0
    
    #widths = list()
    for i in range(len(start)):
        #widths.append(o_end[i, 0] - o_start[i, 0])
        line = LineString([(o_start[i, 0]+basex, o_start[i, 1]+basey),
                           (o_end[i, 0]+basex, o_end[i, 1]+basey)])
        lines.append(line)
    line_id = np.arange(0, len(o_start))
    df = pd.DataFrame(line_id,columns=['id'])
    nids = [nid]*len(o_start)
    df['act_h'] = o_start[:, 2] - info['ref_road']
    df['abs_h'] = o_start[:, 2] + basez
    df['nid'] = np.array(nids)
    df['width'] = np.array(widths)
    gdf = geopandas.GeoDataFrame(df, geometry=lines)
    return gdf

def test(depths, info_path, nid):
    rows, cols = depths.shape
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    #ref_plane = info['ref_plane']
    buttom_edge = info['buttom_edge']
    left_edge = info['left_edge']
    Trans = info['trans_i2o']
    start = np.array([0, 0, 0, 1])
    end = np.array([0.02*cols, 0, 0, 1])
    o_start = Trans @ start.T 
    o_start = o_start.T[0:3]
    o_end = Trans @ end.T
    o_end = o_end.T[0:3]
    basex = info['original_x']
    basey = info['original_y']
    basez = info['original_z']  
    #print(o_start[2], info['ref_road'])
    line = LineString([(o_start[0]+basex, o_start[1]+basey),
                           (o_end[0]+basex, o_end[1]+basey)])
    
    df = pd.DataFrame()
    gdf = geopandas.GeoDataFrame(df, geometry=[line])
    return gdf
    
    
    