# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

#from roi_data_layer.roidb import combined_roidb
#from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
#from tensorboardX import SummaryWriter

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.fpn_net import fpn_net
from model.faster_rcnn.resnet import resnet
import datetime
import pickle

from lib.self_defined_datasets.dataset import CityDataSet
from lib.self_defined_datasets.dataloader import DataLoader
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
#from model.roi_layers import nms
from torchvision.ops.boxes import nms
from model.utils.box_utils import confusion_matrix

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='composite, normal',
                      default='composite_all', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101, fpn',
                    default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=30, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="save",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_false')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str) # sgd, adam
  parser.add_argument('--lr', dest='lr', 
                      help='starting learning rate',
                      default=0.0005, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')


  parser.add_argument('--train_cnn', dest='train_cnn',
                      help='whether pretrain the cnn',
                      action='store_true')
  parser.add_argument('--use_pretrain_cnn', dest='use_pretrain_cnn',
                      help='whether use the pretrained cnn for detection',
                      action='store_true')
  parser.add_argument('--train_stage', dest='train_stage',
                      help='1,2,3,4',
                      default=0, type=int)
  parser.add_argument('--dual', dest='dual',
                      help='whether use dual net',
                      action='store_true')
  args = parser.parse_args()
  return args


def test_epoch(fasterRCNN, val_load, epoch):
    fasterRCNN.eval()
    tps = list()
    fps = list()
    fns = list()
    for idx, blob in enumerate(val_load):
      print('\r{}/{}'.format(idx, len(val_load)), end='')

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(*blob())

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                #box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
                box_deltas = box_deltas.view(1, -1, 4 * 2)

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, blob.im_sizes.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))
      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()

      inds = torch.nonzero(scores[:,1]>cfg.obj_score_thres).view(-1)  #0.5
      if inds.numel() > 0:
        cls_scores = scores[:,1][inds]
        _, order = torch.sort(cls_scores, 0, True)
        if args.class_agnostic:
          cls_boxes = pred_boxes[inds, :]
        else:
          cls_boxes = pred_boxes[inds, 4:]

        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        cls_dets = cls_dets[order]
        keep = nms(cls_boxes[order, :], cls_scores[order], 0.1)
        cls_dets = cls_dets[keep.view(-1).long()]          
        cls_dets = cls_dets.cpu()
        cls_box_dets = cls_dets[:, :-1]
        cls_box_scores = cls_dets[:, -1].numpy()
      else:
        print('  nothing was detected.')
        cls_dets = None
        cls_box_dets = None
      gts_box = blob.gt_boxes.squeeze()[:blob.num_boxes.item(), :-1].cpu()
      tp, fp, fn = confusion_matrix(cls_box_dets, gts_box)
      tps.append(tp)
      fps.append(fp)
      fns.append(fn)
      preds_path = os.path.join(args.save_dir, args.dataset, args.net, 'preds_boxes', 'epoch_{}'.format(-1))
      
      if not os.path.exists(preds_path):
        os.makedirs(preds_path)
      if cls_dets is not None:
        with open(os.path.join(preds_path, '{}.txt'.format(blob.img_names[0])), 'w') as f:
          cls_dets = cls_dets.numpy()*blob.im_sizes[0, 2].item()
          cls_dets[:, -1] /= blob.im_sizes[0, 2].item()
          for i, e in enumerate(cls_dets):
            #f.write(str(cls_box_scores[i])+' '+ ' '.join(map(str, e))+'\n')
            f.write(' '.join(map(str, e))+'\n')

    P = sum(tps)/(sum(tps)+sum(fps)+1e-6)
    R = sum(tps)/(sum(tps)+sum(fns)+1e-6)
    printf('epoch: ', epoch)
    printf("precision:", P)
    printf('recall:', R)
    F1 = (2*P*R)/(P+R+1e-6)
    printf("F1:", F1)
    print("F1:", F1)
    printf('\n\n\n')
    return F1


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def optimistic_restore(network, state_dict, whichs=None):
    mismatch = False
    own_state = network.state_dict()
    if whichs is None:
      for name, param in state_dict.items():
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
            print('matched {}'.format(name))
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True
    else:
      for name, param in state_dict.items():
        if whichs not in name:
          continue
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
            print('matched {}'.format(name))
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("We couldn't find {}".format(','.join(missing)))
        mismatch = True
    return not mismatch


def frozen_layers(network, whichs=''):
  for name, value in dict(network.named_parameters()).items():
    if whichs in name:
      value.requires_grad=False
      #print('frozen {}'.format(name))

if __name__ == '__main__':

  args = parse_args()
  if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)

  std_file_path = os.path.join(args.save_dir, args.dataset, args.net)
  if not os.path.exists(std_file_path):
      os.makedirs(std_file_path)
  stdout = os.path.join(std_file_path, 'exection_{}.log'.format(datetime.date.today()))

  def printf(*data):
      with open(stdout, 'a+') as f:
          print(*data, file=f)

  printf('Called with args:')
  printf(args)

  printf('Using config:')
  pprint.pprint(cfg, stream=open(stdout, 'a+'))
  #np.random.seed(cfg.RNG_SEED)
  #setup_seed(0)
  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    printf("WARNING: You have a CUDA device, so you should probably run with --cuda")

  output_dir = std_file_path

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # train set
  print("dataset: {}".format(args.dataset))
  train_data = CityDataSet(args.dataset, 'val')
  print('train_data: ', len(train_data))

  val_load = DataLoader.loader(train_data, batch_size=args.batch_size, num_workers=1, shuffle=False)
  print('train_loader: ', len(val_load))
  '''
  print("dataset: {}".format(args.dataset))
  train_data = CityDataSet(args.dataset, 'train')
  val_data = CityDataSet(args.dataset,'val')
  print('train_data: ', len(train_data), 'val_data: ', len(val_data))

  dataloader, val_load = DataLoader.splits(train_data, val_data, batch_size=args.batch_size, num_workers=1)
  print('train_loader: ', len(dataloader), 'val_loader: ', len(val_load))
  '''
  #data = next(iter(train_load))
  #print(data.imgs.size(), data.im_sizes.size(),data.gt_boxes.size(),data.num_boxes.size(),)
  train_size = len(train_data)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  
  if args.net == 'res101':
    fasterRCNN = resnet(['bg', 'fg'], 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(['bg', 'fg'], 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(['bg', 'fg'], 152, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'fpn50':
    fasterRCNN = fpn_net(['bg', 'fg'], 50, class_agnostic=args.class_agnostic)
  elif args.net == 'fpn101':
    fasterRCNN = fpn_net(['bg', 'fg'], 101, class_agnostic=args.class_agnostic)
  else:
    printf("network is not defined")
    pdb.set_trace()
  
  fasterRCNN.create_architecture()

  if args.cuda:
    fasterRCNN.cuda()

  load_name = os.path.join(output_dir, 'faster_rcnn_best.pth')
  print(load_name)
  with open(load_name, 'rb') as f:
    checkpoint = pickle.load(f)
  fasterRCNN.load_state_dict(checkpoint['model'])

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  performace = test_epoch(fasterRCNN, val_load, 0)
