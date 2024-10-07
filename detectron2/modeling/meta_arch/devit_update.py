# Copyright (c) Facebook, Inc. and its affiliates.
import math
import random
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.lib import pad
import torch
from torch import nn
from torch.nn import functional as F
from random import randint
from torch.cuda.amp import autocast


from detectron2.config import configurable, get_cfg
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
import warnings
from detectron2.data.datasets.coco_zeroshot_categories import COCO_SEEN_CLS, \
    COCO_UNSEEN_CLS, COCO_OVD_ALL_CLS
from ..roi_heads import build_roi_heads
from ..matcher import Matcher
from .build import META_ARCH_REGISTRY


from PIL import Image
import copy
from ..backbone.fpn import build_resnet_fpn_backbone
from detectron2.utils.comm import gather_tensors, MILCrossEntropy

from detectron2.layers.roi_align import ROIAlign
from torchvision.ops.boxes import box_area, box_iou

from torchvision.ops import sigmoid_focal_loss

from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.structures.masks import PolygonMasks

from lib.dinov2.layers.block import Block
from lib.regionprop_update import augment_rois, region_coord_2_abs_coord, abs_coord_2_region_coord, SpatialIntegral, RegionPropagationNetwork
from lib.categories import SEEN_CLS_DICT, ALL_CLS_DICT


####################################################################
# LOSS
####################################################################


def generalized_box_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/

    The input boxes should be in (x0, y0, x1, y1) format

    Args:
        boxes1: (torch.Tensor[N, 4]): first set of boxes
        boxes2: (torch.Tensor[M, 4]): second set of boxes

    Returns:
        torch.Tensor: a NxM pairwise matrix containing the pairwise Generalized IoU
        for every element in boxes1 and boxes2.
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = elementwise_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def focal_loss(inputs, targets, gamma=0.5, reduction="mean", bg_weight=0.2, num_classes=None):
    """Inspired by RetinaNet implementation"""
    if targets.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    
    # focal scaling
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    p = F.softmax(inputs, dim=-1)
    p_t = p[torch.arange(p.size(0)).to(p.device), targets]  # get prob of target class
    p_t = torch.clamp(p_t, 1e-7, 1-1e-7) # prevent NaN
    loss = ce_loss * ((1 - p_t) ** gamma)

    # bg loss weight
    if bg_weight >= 0:
        assert num_classes is not None
        loss_weight = torch.ones(loss.size(0)).to(p.device)
        loss_weight[targets == num_classes] = bg_weight
        loss = loss * loss_weight

    if reduction == "mean":
        loss = loss.mean()

    return loss

####################################################################
# Operation
####################################################################


def elementwise_box_iou(boxes1, boxes2) -> Tuple[torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def distance_embed(x, temperature = 10000, num_pos_feats = 128, scale=10.0):
    # x: [bs, n_dist]
    x = x[..., None]
    scale = 2 * math.pi * scale
    dim_t = torch.arange(num_pos_feats)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    sin_x = x * scale / dim_t.to(x.device)
    emb = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
    return emb # [bs, n_dist, n_emb]


def interpolate(seq, T, mode='linear', force=False):
    # seq: B x C x L
    if (seq.shape[-1] < T) or force:
        return F.interpolate(seq, T, mode=mode) 
    else:
    #     # assume is sorted ascending order
        return seq[:, :, -T:]

def box_cxcywh_to_xyxy(bbox) -> torch.Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.unbind(-1)
    new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(new_bbox, dim=-1)


def box_xyxy_to_cxcywh(bbox) -> torch.Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    x0, y0, x1, y1 = bbox.unbind(-1)
    new_bbox = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(new_bbox, dim=-1)



####################################################################
# MISC 
####################################################################

def _log_classification_stats(pred_logits, gt_classes):
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    try:
        storage = get_event_storage()
        storage.put_scalar(f"cls_acc", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar(f"fg_cls_acc", fg_num_accurate / num_fg)
            storage.put_scalar(f"false_neg_ratio", num_false_negative / num_fg)
    except:
        pass


####################################################################
# Main Class
####################################################################


@META_ARCH_REGISTRY.register()
class OpenSetDetectorWithExamples_refactored(nn.Module):

    @property
    def device(self):
        return self.pixel_mean.device

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [((x / 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(
            instances, batched_inputs):
            height = input_per_image["height"]  # original image size, before resizing
            width = input_per_image["width"]  # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    @configurable
    def __init__(self,
                offline_backbone: Backbone,
                backbone: Backbone,
                offline_proposal_generator: nn.Module, 

                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],

                offline_pixel_mean: Tuple[float],
                offline_pixel_std: Tuple[float],
                offline_input_format: Optional[str] = None,

                class_prototypes_file="",
                bg_prototypes_file="",
                roialign_size=14,
                box_noise_scale=1.0,
                proposal_matcher = None,

                box2box_transform=None,
                smooth_l1_beta=0.0,
                test_score_thresh=0.001,
                test_nms_thresh=0.5,
                test_topk_per_image=100,
                cls_temp=0.1,
                
                num_sample_class=-1,
                seen_cids = [],
                all_cids = [],
                T_length=128,

                input_feat_dim=1024,
                proj_feat_dim=256,
                pos_emb_size=128,
                dist_emb_size=128,
                hidden_size=256,
                num_layers=4,
                
                bg_cls_weight=0.2,
                batch_size_per_image=128,
                pos_ratio=0.25,
                mult_rpn_score=False,
                ):
        super().__init__()
        if ',' in class_prototypes_file:
            class_prototypes_file = class_prototypes_file.split(',')
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.backbone = backbone 
        self.bg_cls_weight = bg_cls_weight
        self.box2box_transform = box2box_transform 

        if np.sum(pixel_mean) < 3.0: # convert pixel value to range [0.0, 1.0] by dividing 255.0
            self.div_pixel = True
        else:
            self.div_pixel = False

        # RPN related 
        self.input_format = "RGB"
        self.offline_backbone = offline_backbone
        self.offline_proposal_generator = offline_proposal_generator        
        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else:
                self.offline_div_pixel = False
        
        self.proposal_matcher = proposal_matcher
        
        # class_prototypes_file
        #  prototypes, class_order_for_inference
        if isinstance(class_prototypes_file, str):
            dct = torch.load(class_prototypes_file)
            prototypes = dct['prototypes']
            if 'label_names' not in dct:
                warnings.warn("label_names not found in class_prototypes_file, using COCO_SEEN_CLS + COCO_UNSEEN_CLS")
                prototype_label_names = COCO_SEEN_CLS + COCO_UNSEEN_CLS
                assert len(prototype_label_names) == len(prototypes)
            else:
                prototype_label_names = dct['label_names']
        elif isinstance(class_prototypes_file, list):
            p1, p2 = torch.load(class_prototypes_file[0]), torch.load(class_prototypes_file[1])
            prototypes = torch.cat([p1['prototypes'], p2['prototypes']], dim=0)
            prototype_label_names = p1['label_names'] + p2['label_names']
        else:
            raise NotImplementedError()

        if len(prototypes.shape) == 3:
            class_weights = F.normalize(prototypes.mean(dim=1), dim=-1)
        else:
            class_weights = F.normalize(prototypes, dim=-1)
        
        self.num_train_classes = len(seen_cids)
        self.num_classes = len(all_cids)

        train_class_order = [prototype_label_names.index(c) for c in seen_cids]
        test_class_order = [prototype_label_names.index(c) for c in all_cids]

        self.label_names = prototype_label_names
        assert -1 not in train_class_order and -1 not in test_class_order

        self.register_buffer("train_class_weight", class_weights[torch.as_tensor(train_class_order)])
        self.register_buffer("test_class_weight", class_weights[torch.as_tensor(test_class_order)])
        self.test_class_order = test_class_order

        self.all_labels = all_cids
        self.seen_labels = seen_cids

        bg_protos = torch.load(bg_prototypes_file)
        if isinstance(bg_protos, dict):  # NOTE: connect to dict output of `generate_prototypes`
            bg_protos = bg_protos['prototypes']
        if len(bg_protos.shape) == 3:
            bg_protos = bg_protos.flatten(0, 1)
        self.register_buffer("bg_tokens", bg_protos)
        self.num_bg_tokens = len(self.bg_tokens)

        self.roialign_size = roialign_size
        self.roi_align_77 = ROIAlign(7, 1 / backbone.patch_size, sampling_ratio=-1)
        self.roi_align = ROIAlign(roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)
        # input: NCHW, Bx5, output BCKK
        self.box_noise_scale = box_noise_scale
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

        self.T = T_length
        self.input_feat_dim = input_feat_dim
        self.proj_feat_dim = proj_feat_dim
        self.pos_emb_size = pos_emb_size
        self.dist_emb_size = dist_emb_size
        self.hidden_size = hidden_size

        self.foreground_linears = nn.ModuleDict({
            'current_class': nn.Linear(pos_emb_size, dist_emb_size),
            'other_classes': nn.Linear(T_length, dist_emb_size),
            'background': nn.Linear(self.num_bg_tokens, pos_emb_size),
            'feat': nn.Linear(input_feat_dim, proj_feat_dim)
        })
        self.background_linears = nn.ModuleDict({
            'classes': nn.Linear(T_length, dist_emb_size),
            'background': nn.Linear(self.num_bg_tokens, pos_emb_size),
            'feat': nn.Linear(input_feat_dim, proj_feat_dim)
        })       
        self.rpropnet = RegionPropagationNetwork(num_layers, proj_feat_dim + 2*dist_emb_size + pos_emb_size, 
                                                hidden_size, roialign_size)
        self.rpropnet_bg = RegionPropagationNetwork(num_layers, proj_feat_dim + dist_emb_size + pos_emb_size, 
                                                    hidden_size, roialign_size, classification_only=True)
        
        self.cls_temp = cls_temp
        self.num_sample_class = num_sample_class
        self.batch_size_per_image = batch_size_per_image
        self.pos_ratio = pos_ratio
        self.mult_rpn_score = mult_rpn_score


    @classmethod
    def from_config(cls, cfg, use_bn=False):
        offline_cfg = get_cfg()
        offline_cfg.merge_from_file(cfg.DE.OFFLINE_RPN_CONFIG)
        if cfg.DE.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
            offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
            offline_cfg.MODEL.RESNETS.NORM = "BN" # 5 resnet layers
            offline_cfg.MODEL.FPN.NORM = "BN" # fpn layers
            # offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
            # offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
            offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
        if cfg.DE.OFFLINE_RPN_NMS_THRESH:
            offline_cfg.MODEL.RPN.NMS_THRESH = cfg.DE.OFFLINE_RPN_NMS_THRESH  # 0.9
        if cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST:
            offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST # 1000

        # create offline backbone and RPN
        offline_backbone = build_backbone(offline_cfg)
        offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())

        # convert to evaluation mode
        for p in offline_backbone.parameters(): p.requires_grad = False
        for p in offline_rpn.parameters(): p.requires_grad = False
        offline_backbone.eval()
        offline_rpn.eval()

        backbone = build_backbone(cfg)
        for p in backbone.parameters(): p.requires_grad = False
        backbone.eval()

        feat_dims = {'large': 1024, 'base': 768, 'small': 384}

        return {
            "backbone": backbone,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "class_prototypes_file": cfg.DE.CLASS_PROTOTYPES,
            "bg_prototypes_file": cfg.DE.BG_PROTOTYPES,

            "roialign_size": cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,

            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
            "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
            "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
            
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),

            # regression
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,

            "box_noise_scale": 0.5,
            "cls_temp": cfg.DE.TEMP,
            "num_sample_class": cfg.DE.TOPK,
            
            "seen_cids": SEEN_CLS_DICT[cfg.DATASETS.TRAIN[0]],
            "all_cids": ALL_CLS_DICT[cfg.DATASETS.TRAIN[0]],
            "T_length": cfg.DE.T,

            "input_feat_dim": feat_dims[cfg.MODEL.BACKBONE.TYPE],
            "proj_feat_dim": 256,
            "pos_emb_size": 128,
            "dist_emb_size": 64,
            "hidden_size": 256,
            
            "bg_cls_weight": cfg.DE.BG_CLS_LOSS_WEIGHT,
            "batch_size_per_image": cfg.DE.RCNN_BATCH_SIZE,
            "pos_ratio": cfg.DE.POS_RATIO,
            
            "mult_rpn_score": cfg.DE.MULTIPLY_RPN_SCORE,
            "num_layers": cfg.DE.NUM_CLS_LAYERS,
        }
    
    def prepare_noisy_boxes(self, gt_boxes, image_shape):
        noisy_boxes = []

        H, W = image_shape[2:]
        H, W = float(H), float(W)

        for box in gt_boxes:
            box = box.repeat(5, 1) # duplicate more noisy boxes
            box_ccwh = box_xyxy_to_cxcywh(box) 

            diff = torch.zeros_like(box_ccwh)
            diff[:, :2] = box_ccwh[:, 2:] / 2
            diff[:, 2:] = box_ccwh[:, 2:] / 2

            rand_sign = (
                torch.randint_like(box_ccwh, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            ) 
            rand_part = torch.rand_like(box_ccwh) * rand_sign
            box_ccwh = box_ccwh + torch.mul(rand_part, diff).to(box_ccwh.device) * self.box_noise_scale

            noisy_box = box_cxcywh_to_xyxy(box_ccwh)

            noisy_box[:, 0].clamp_(min=0.0, max=W)
            noisy_box[:, 2].clamp_(min=0.0, max=W)
            noisy_box[:, 1].clamp_(min=0.0, max=H)
            noisy_box[:, 3].clamp_(min=0.0, max=H)

            noisy_boxes.append(noisy_box)

        return noisy_boxes
    
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        bs = len(batched_inputs)
        loss_dict = {}
        if not self.training: assert bs == 1
        class_weights = self.train_class_weight if self.training else  self.test_class_weight
        num_classes = len(class_weights)

        with torch.no_grad():
            # with autocast(enabled=True):
            if self.offline_backbone.training or self.offline_proposal_generator.training:  
                self.offline_backbone.eval() 
                self.offline_proposal_generator.eval()  
            images = self.offline_preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            proposals, _ = self.offline_proposal_generator(images, features, None)     
            images = self.preprocess_image(batched_inputs)
            H, W = images.tensor.shape[2:]
        
        with torch.no_grad():
            if self.backbone.training: self.backbone.eval()
            with autocast(enabled=True):
                all_patch_tokens = self.backbone(images.tensor)
            output_key = sorted(list(all_patch_tokens.keys()), key=lambda x: int(x[3:]))[-1]
            patch_tokens = all_patch_tokens[output_key]

        if self.training: 
            with torch.no_grad():
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_boxes = [x.gt_boxes.tensor for x in gt_instances]
                rpn_boxes = [x.proposal_boxes.tensor for x in proposals]

                noisy_boxes = self.prepare_noisy_boxes(gt_boxes, images.tensor.shape)
                boxes = [torch.cat([gt_boxes[i], noisy_boxes[i], rpn_boxes[i]]) 
                        for i in range(len(batched_inputs))]

                class_labels = []
                matched_gt_boxes = []
                resampled_proposals = []

                num_bg_samples, num_fg_samples = [], []

                for proposals_per_image, targets_per_image in zip(boxes, gt_instances):
                    match_quality_matrix = box_iou(
                        targets_per_image.gt_boxes.tensor, proposals_per_image
                    ) # (N, M)
                    matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                    if len(targets_per_image.gt_classes) > 0:
                        class_labels_i = targets_per_image.gt_classes[matched_idxs]
                    else:
                        # no annotation on this image
                        assert torch.all(matched_labels == 0)
                        class_labels_i = torch.zeros_like(matched_idxs)
                    class_labels_i[matched_labels == 0] = num_classes
                    class_labels_i[matched_labels == -1] = -1
                    
                    if self.training or self.evaluation_shortcut:
                        positive = ((class_labels_i != -1) & (class_labels_i != num_classes)).nonzero().flatten()
                        negative = (class_labels_i == num_classes).nonzero().flatten()

                        batch_size_per_image = self.batch_size_per_image # 512
                        num_pos = int(batch_size_per_image * self.pos_ratio)
                        # protect against not enough positive examples
                        num_pos = min(positive.numel(), num_pos)
                        num_neg = batch_size_per_image - num_pos
                        # protect against not enough negative examples
                        num_neg = min(negative.numel(), num_neg)

                        perm1 = torch.randperm(positive.numel(), device=self.device)[:num_pos]
                        perm2 = torch.randperm(negative.numel())[:num_neg].to(self.device) # torch.randperm(negative.numel(), device=negative.device)[:num_neg]
                        pos_idx = positive[perm1]
                        neg_idx = negative[perm2]
                        sampled_idxs = torch.cat([pos_idx, neg_idx], dim=0)
                    else:
                        sampled_idxs = torch.arange(len(proposals_per_image), device=self.device).long()

                    proposals_per_image = proposals_per_image[sampled_idxs]
                    class_labels_i = class_labels_i[sampled_idxs]
                    
                    if len(targets_per_image.gt_boxes.tensor) > 0:
                        gt_boxes_i = targets_per_image.gt_boxes.tensor[matched_idxs[sampled_idxs]]
                    else:
                        gt_boxes_i = torch.zeros(len(sampled_idxs), 4, device=self.device) # not used anyway

                    resampled_proposals.append(proposals_per_image)
                    class_labels.append(class_labels_i)
                    matched_gt_boxes.append(gt_boxes_i)

                    num_bg_samples.append((class_labels_i == num_classes).sum().item())
                    num_fg_samples.append(class_labels_i.numel() - num_bg_samples[-1])
                

                class_labels = torch.cat(class_labels)
                matched_gt_boxes = torch.cat(matched_gt_boxes) # for regression purpose.
                
                rois = []
                for bid, box in enumerate(resampled_proposals):
                    batch_index = torch.full((len(box), 1), fill_value=float(bid)).to(self.device) 
                    rois.append(torch.cat([batch_index, box], dim=1))
                rois = torch.cat(rois)
        else:
            boxes = proposals[0].proposal_boxes.tensor 
            rois = torch.cat([torch.full((len(boxes), 1), fill_value=0).to(self.device) , 
                            boxes], dim=1)

        if self.training:
            fg_indices = class_labels != num_classes 
            bg_indices = class_labels == num_classes
            matched_gt_boxes[bg_indices] = rois[bg_indices, 1:] # nx4
            aug_rois, init_region, gt_region, _ = augment_rois(rois[:, 1:], matched_gt_boxes, img_h=H, img_w=W, pooler_size=self.roialign_size, 
                        min_expansion=0.4, expand_shortest=True)
            aug_rois = torch.cat([rois[:, :1], aug_rois], dim=1)
            gt_region_coords = abs_coord_2_region_coord(aug_rois[:, 1:], matched_gt_boxes, self.roialign_size)
        else:
            aug_rois, init_region, _, _ = augment_rois(rois[:, 1:], None, img_h=H, img_w=W, pooler_size=self.roialign_size, 
                        min_expansion=0.4, expand_shortest=True)
            aug_rois = torch.cat([rois[:, :1], aug_rois], dim=1)


        roi_features_origin = self.roi_align_77(patch_tokens, rois)
        roi_features = self.roi_align(patch_tokens, aug_rois) # N, C, k, k
        roi_bs = len(roi_features)

        # roi_features # N x emb x spatial
        roi_features_origin = roi_features_origin.flatten(2)
        roi_features = roi_features.flatten(2) 
        bs, spatial_size = roi_features.shape[0], roi_features.shape[-1]
        # (N x spatial x emb) @ (emb x class) = N x spatial x class
        feats = roi_features.transpose(-2, -1) @ class_weights.T

        num_active_classes = self.num_sample_class
        sample_class_enabled = True

        init_scores = F.normalize(roi_features_origin.flatten(2).mean(2), dim=1) @ class_weights.T
        topk_class_indices = torch.topk(init_scores, num_active_classes, dim=1).indices

        if self.training:
            class_indices = []
            for i in range(roi_bs):
                curr_label = class_labels[i].item()
                topk_class_indices_i = topk_class_indices[i].cpu()
                if curr_label in topk_class_indices_i or curr_label == num_classes:
                    curr_indices = topk_class_indices_i
                else:
                    curr_indices = torch.cat([torch.as_tensor([curr_label]), 
                                        topk_class_indices_i[:-1]])
                class_indices.append(curr_indices)
            class_indices = torch.stack(class_indices).to(self.device) 
        else:
            class_indices = topk_class_indices
        
        class_indices = torch.sort(class_indices, dim=1).values

        # Class Shuffle
        other_classes = [] 
        indexes = torch.arange(0, num_classes, device=self.device)[None, None, :].repeat(bs, spatial_size, 1)
        for i in range(num_active_classes):
            cmask = indexes != class_indices[:, i].view(-1, 1, 1)
            _ = torch.gather(feats, 2, indexes[cmask].view(bs, spatial_size, num_classes - 1)) # N x spatial x classes-1
            other_classes.append(_[:, :, None, :]) 

        other_classes = torch.cat(other_classes, dim=2)  # N x spatial x classes x classes-1
        other_classes = other_classes.permute(0, 2, 1, 3) # N x classes x spatial x classes-1
        other_classes = other_classes.flatten(0, 1) # (Nxclasses) x spatial x classes-1
        other_classes, _ = torch.sort(other_classes, dim=-1)
        other_classes = interpolate(other_classes, self.T, mode='linear') # (Nxclasses) x spatial x T
        other_classes = self.foreground_linears['other_classes'](other_classes) # (Nxclasses) x spatial x emb
        other_classes = other_classes.permute(0, 2, 1) # (Nxclasses) x emb x spatial
        # (Nxclasses) x emb x S x S
        inter_dist_emb = other_classes.reshape(bs * num_active_classes, -1, self.roialign_size, self.roialign_size)

        intra_feats = torch.gather(feats, 2, class_indices[:, None, :].repeat(1, spatial_size, 1))
        intra_dist_emb = distance_embed(intra_feats.flatten(0, 1), num_pos_feats=self.pos_emb_size) # (Nxspatial) x class x emb TODO: 可以再加一个 linear 层这里
        intra_dist_emb = self.foreground_linears['current_class'](intra_dist_emb)
        intra_dist_emb = intra_dist_emb.reshape(bs, spatial_size, num_active_classes, -1)

        # (Nxclasses) x emb x S x S
        intra_dist_emb = intra_dist_emb.permute(0, 2, 3, 1).flatten(0, 1).reshape(bs * num_active_classes, -1, 
                                                                                self.roialign_size, self.roialign_size)

        bg_feats = roi_features.transpose(-2, -1) @ self.bg_tokens.T # N x spatial x back
        bg_dist_emb = self.foreground_linears['background'](bg_feats) # N x spatial x emb
        bg_dist_emb = bg_dist_emb.permute(0, 2, 1).reshape(bs, -1, self.roialign_size, self.roialign_size)
        bg_dist_emb_c = bg_dist_emb[:, None, :, :, :].expand(-1, num_active_classes, -1, -1, -1).flatten(0, 1)

        projected_feats = self.foreground_linears['feat'](roi_features.transpose(-2, -1)).permute(0, 2, 1).reshape(bs, -1, self.roialign_size, self.roialign_size)

        # (Nxclasses) x EMB x S x S
        fg_x = torch.cat([intra_dist_emb, inter_dist_emb, bg_dist_emb_c, 
                                    projected_feats[:, None, :, :, :].expand(-1, num_active_classes, -1, -1, -1).flatten(0, 1)], dim=1)

        cls_dist_feats = interpolate(torch.sort(feats, dim=2).values, self.T, mode='linear') # N x spatial x T
        bg_cls_dist_emb = self.background_linears['classes'](cls_dist_feats) # N x spatial x emb
        bg_cls_dist_emb = bg_cls_dist_emb.permute(0, 2, 1).reshape(bs, -1, self.roialign_size, self.roialign_size)
        bg_dist_emb = self.background_linears['background'](bg_feats) # N x spatial x emb
        bg_dist_emb = bg_dist_emb.permute(0, 2, 1).reshape(bs, -1, self.roialign_size, self.roialign_size)
        projected_feats = self.background_linears['feat'](roi_features.transpose(-2, -1)).permute(0, 2, 1).reshape(bs, -1, self.roialign_size, self.roialign_size)
        
        bg_x = torch.cat([bg_cls_dist_emb, bg_dist_emb, projected_feats], dim=1)

        fg_output = self.rpropnet(fg_x, init_region[:, None, :, :].repeat(num_active_classes, 1, 1, 1)) # TODO: init_region needs repeat
        bg_output = self.rpropnet_bg(bg_x, init_region[:, None, :, :])

        if self.training:
            logits = []
            gt_region = gt_region[fg_indices].float().flatten(1)
            gt_region_coords = gt_region_coords[fg_indices]
            num_masks = fg_indices.sum()
            fg_indices_int = fg_indices.nonzero().flatten()
            fg_class_indices_int = (class_labels[fg_indices][:, None] == class_indices[fg_indices]).nonzero()[:, 1]
            for layer_id, (fgo, bgo) in enumerate(zip(fg_output, bg_output)):
                c, b = fgo['class_score'].reshape(bs, num_active_classes), bgo['class_score']
                logits.append(torch.cat([c, b], dim=1) / self.cls_temp)

                pred_region_coords = fgo['box_coords'].reshape(bs, num_active_classes, 4)
                pred_mask_logits = fgo['output_region'].reshape(bs, num_active_classes, self.roialign_size, self.roialign_size)
                pred_region_coords = pred_region_coords[fg_indices_int, fg_class_indices_int]
                pred_mask_logits = pred_mask_logits[fg_indices_int, fg_class_indices_int]

                loss_dict[f"region_bce_loss_{layer_id}"] = sigmoid_ce_loss(pred_mask_logits.flatten(1), gt_region, num_masks)
                loss_dict[f"region_dice_loss_{layer_id}"] = dice_loss(pred_mask_logits.flatten(1), gt_region, num_masks)
                loss_dict[f'rg_l1_loss_{layer_id}'] = F.l1_loss(pred_region_coords, gt_region_coords)
                try:
                    loss_dict[f'rg_giou_loss_{layer_id}'] = (1 - torch.diag(generalized_box_iou(
                                    box_cxcywh_to_xyxy(pred_region_coords),
                                    box_cxcywh_to_xyxy(gt_region_coords)))).mean()
                except:
                    pass

        else:
            cls_logits = fg_output[-1]['class_score'].reshape(bs, num_active_classes)
            bg_logits = bg_output[-1]['class_score']
            logits = torch.cat([cls_logits, bg_logits], dim=1)
            logits = logits / self.cls_temp


        aug_rois = aug_rois[:, None, :].repeat(1, num_active_classes, 1).flatten(0, 1)
        pred_region_coords = fg_output[-1]['box_coords']
        pred_abs_boxes = region_coord_2_abs_coord(aug_rois[:, 1:], pred_region_coords, self.roialign_size)
        if self.training:
            matched_gt_boxes = matched_gt_boxes[fg_indices] # nx4
            fg_proposals = rois[fg_indices, 1:] # nx4
            pred_abs_boxes = pred_abs_boxes.reshape(-1, num_active_classes, 4)
            pred_abs_boxes = pred_abs_boxes[fg_indices_int, fg_class_indices_int]

            fg_pred_deltas =  self.box2box_transform.get_deltas(fg_proposals, pred_abs_boxes)
        else:
            pred_deltas = self.box2box_transform.get_deltas(rois[:, None, 1:].repeat(1, num_active_classes, 1).flatten(0, 1), pred_abs_boxes)
            pred_deltas = pred_deltas.reshape(bs, num_active_classes, 4)
            pred_deltas = pred_deltas.flatten(1)


        if self.training:
            class_labels = class_labels.long()
            bg_indices = class_labels == num_classes
            fg_indices = class_labels != num_classes

            class_labels[fg_indices] = (class_indices == class_labels.view(-1, 1)).nonzero()[:, 1]
            class_labels[bg_indices] = num_active_classes           

            _log_classification_stats(logits[-1].detach(), class_labels)

            for i, l in enumerate(logits):
                loss = focal_loss(l, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight)
                loss_dict[f'focal_loss_{i}'] = loss
    

            gt_pred_deltas = self.box2box_transform.get_deltas(
                fg_proposals,
                matched_gt_boxes,
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )

            box_loss = loss_box_reg / max(class_labels.numel(), 1.0)
            if not torch.isinf(box_loss).any():
                loss_dict['bbox_loss'] = box_loss
            else:
                loss_dict['bbox_loss'] = torch.zeros(1, device=self.device)
                
            return loss_dict
        else:
            assert len(proposals) == 1
            image_size = proposals[0].image_size

            scores = F.softmax(logits, dim=-1)
            output = {'scores': scores[:, :-1] }

            predict_boxes = self.box2box_transform.apply_deltas(
                pred_deltas, # N, k*4
                rois[:, 1:],
            )

            full_scores = torch.zeros(len(scores), num_classes + 1, device=self.device)
            full_scores.scatter_(1, class_indices, scores)
            full_scores[:, -1] = scores[:, -1]

            full_boxes = torch.zeros(len(scores), num_classes, 4, device=self.device)
            predict_boxes = predict_boxes.view(len(scores), num_active_classes, 4)
            full_boxes.scatter_(1, class_indices[:, :, None].repeat(1, 1, 4), predict_boxes)
            full_boxes = full_boxes.flatten(1)

            # re-assign back
            scores = full_scores
            output['scores'] = full_scores[:, :-1]
            predict_boxes = full_boxes
            
            
            if self.mult_rpn_score:
                rpn_scores = [x.objectness_logits for x in proposals][0]
                rpn_scores[rpn_scores < 0] = 0
                scores = (scores * rpn_scores[:, None]) ** 0.5
            
            instances, _ = fast_rcnn_inference(
                    [predict_boxes],
                    [scores],
                    [image_size],
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    False,
                    "gaussian",
                    0.5,
                    0.01,
                    self.test_topk_per_image,
                    scores_bf_multiply = [scores],
                    vis = False
                ) 

            results = self._postprocess(instances, batched_inputs)
            output['instances'] = results[0]['instances']
            return [output, ]