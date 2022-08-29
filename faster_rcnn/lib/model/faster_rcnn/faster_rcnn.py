import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.rpn.anchor_target_layer import _AnchorTargetLayer
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw
from torchvision.ops import RoIAlign
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict

#from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from .anchor_utils import AnchorGenerator, ImageList

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, multi_feat=False,
        rpn_anchor_generator=None, rpn_head=None,
        rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5):

        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.multi_feat = multi_feat

        if multi_feat:
            anchor_sizes = ((8, 16), (16, 32), (32,64), (64,128), (128,256))
            aspect_ratios = ((0.33, 0.5, 1.0, 2.0, 3.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        else:
            rpn_anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128, 256),),
                                   aspect_ratios=((0.33, 0.5, 1.0, 2.0, 3.0),))

        if rpn_head is None:
            rpn_head = RPNHead(
                self.dout_base_model, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.RCNN_rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # define rpn
        #self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        #self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        #self.RCNN_roi_align = RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        self.RCNN_roi_align = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'] if self.multi_feat else ['0'],
                output_size=cfg.POOLING_SIZE,
                sampling_ratio=2)

    def forward(self, im_data, im_data_2, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        if im_info is not None:
          im_info = im_info.data
        if gt_boxes is not None:
          gt_boxes = gt_boxes.data
        if num_boxes is not None:
          num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        if isinstance(base_feat, torch.Tensor):
            base_feat = OrderedDict([('0', base_feat)])
        
        # feed base feature map tp RPN to obtain rois
        if self.training:
            targets = list()
            for i, e in enumerate(num_boxes):
                target = dict()
                target['boxes'] = gt_boxes[i, :e][:, :-1]
                target['labels'] = gt_boxes[i, :e][:, -1].long()
                target['image_id'] = gt_boxes.new(1,).fill_(i).long()
                targets.append(target)

        im_data_list = ImageList(im_data, [(im_data.size(2), im_data.size(3))])
        #rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        
        rpn_boxes, rpn_losses = self.RCNN_rpn(im_data_list, base_feat, targets=targets if self.training else None)
        #print(rpn_boxes[0].size())  # list ([2000, 4], ... )
        rois = list()
        for i in range(len(rpn_boxes)):
            box = rpn_boxes[i]
            inds = box.new(box.size(0), 1).fill_(i)
            roi_per_img = torch.cat([inds, box], dim=1)
            rois.append(roi_per_img)
        rois = torch.stack(rois, dim=0)
        
        # rois: [b, post_nms_topN, 5]
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            rpn_loss_cls, rpn_loss_bbox = rpn_losses['loss_objectness'], rpn_losses['loss_rpn_box_reg']
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        #print(rois.size())
        #raise SystemExit()
        # do roi pooling based on predicted rois
        proposals = list()
        image_shapes = list()
        for i in range(rois.size(0)):
            proposals.append(rois[i, :, 1:])
            image_shapes.append((cfg.IM_SCALE, cfg.IM_SCALE))
        
        pooled_feat = self.RCNN_roi_align(base_feat, proposals, image_shapes)
        #pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        #normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def generate_samples(self, batch_size, size_info, gt_boxes, im_info, num_boxes):
        RPN_anchor_target = _AnchorTargetLayer(cfg.FEAT_STRIDE[0], cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS)
        rpn_data = RPN_anchor_target((size_info, gt_boxes, im_info, num_boxes))
        bg_bboxes = rpn_data[-1]  # [b, 128, 4]
        if not self.training:
            num_gt = num_boxes[0].cpu().item()
            bg_box = bg_bboxes[0]    
            num_bg = bg_box.size(0)

            if num_bg <= num_gt:
              box_buffer = torch.cat((bg_box, gt_boxes[0, :num_gt, :4]), dim=0)
            else:
              box_buffer = torch.cat((bg_box[:num_gt], gt_boxes[0, :num_gt, :4]), dim=0)
            inds = box_buffer.new(box_buffer.size(0), 1).fill_(0).float()
            box_buffer = torch.cat((inds, box_buffer), dim=1)
            label_buffer = box_buffer.new(1, box_buffer.size(0)).fill_(0).long()
            label_buffer[0, -num_gt:] = label_buffer.new(num_gt, ).fill_(1)
            box_buffer = box_buffer.unsqueeze(0)

            return label_buffer, box_buffer

        batch_per_img = bg_bboxes.size(1)
        box_buffer = bg_bboxes.new(batch_size, batch_per_img, 5).zero_()
        label_buffer = bg_bboxes.new(batch_size, batch_per_img).zero_().long()

        for i in range(batch_size):
            num_gt = num_boxes[i].cpu().item()
            bg_box = bg_bboxes[i]
            tmp_num = min(batch_per_img, num_gt) 
            box_buffer[:, :, 0] = box_buffer.new(batch_per_img, ).fill_(i).float()
            box_buffer[i, :-tmp_num, 1:] = bg_box[:-tmp_num]
            box_buffer[i, -tmp_num:, 1:] = gt_boxes[i, :tmp_num, :4]
            label_buffer[i, -tmp_num:] = label_buffer.new(tmp_num, ).fill_(1)
            rand_inds = torch.from_numpy(np.random.permutation(batch_per_img)).type_as(label_buffer).long()
            box_buffer[i] = box_buffer[i][rand_inds]
            label_buffer[i] = label_buffer[i][rand_inds]
        return label_buffer, box_buffer
