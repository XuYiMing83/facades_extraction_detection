from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
import pickle
from torchvision.models.detection import FasterRCNN
from .resnet import resnet101


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
}

class fpn_net(_fasterRCNN):
  def __init__(self, classes, num_layers=50, pretrained=True, class_agnostic=False,
  				trainable_backbone_layers=3, pretrained_backbone=False):
    self.dout_base_model = 256
    self.classes = classes
    self.pretrained = pretrained
    self.num_layers = num_layers
    self.class_agnostic = class_agnostic
    self.trainable_backbone_layers = trainable_backbone_layers
    self.pretrained_backbone = pretrained_backbone
    self.representation_size = 2048
    _fasterRCNN.__init__(self, classes, class_agnostic, multi_feat=True)

  def _init_modules(self):
    #resnet = resnet101(pretrained=True)
    if self.pretrained:
        kwargs = {}
        if self.num_layers == 50:
            backbone = resnet_fpn_backbone('resnet50', pretrained=False)
            model = FasterRCNN(backbone, 91, **kwargs)
            state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                                  progress=True)
            model.load_state_dict(state_dict)
            self.RCNN_base = model.backbone
        elif self.num_layers == 101:
            res101_bone = resnet101(pretrained=True)
            backbone = resnet_fpn_backbone('resnet101', pretrained=True)
            state = load_state_dict_from_url(model_urls['resnet101'],
                                                  progress=True)
            optimistic_restore(backbone.body, state)
            self.RCNN_base = backbone
        else:
            raise NotImplementedError
    else:
        self.RCNN_base = resnet_fpn_backbone('resnet50', pretrained=True)
    self.RCNN_top = TwoMLPHead(
                	self.dout_base_model * cfg.POOLING_SIZE ** 2,
                	self.representation_size)

    self.RCNN_cls_score = nn.Sequential(
      nn.Linear(self.representation_size, 2048), 
      nn.ReLU(), 
      nn.Linear(2048, 2048), 
      nn.Linear(2048, self.n_classes)
    )
    
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Sequential(
      nn.Linear(self.representation_size, 2048), 
      nn.ReLU(), 
      nn.Linear(2048, 2048), 
      nn.Linear(2048, 4)
    )
    else:
      self.RCNN_bbox_pred = nn.Sequential(
      nn.Linear(self.representation_size, 2048), 
      nn.ReLU(), 
      nn.Linear(2048, 2048), 
      nn.Linear(2048, 4 * self.n_classes)
    )
  
  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5)
    return fc7


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

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
