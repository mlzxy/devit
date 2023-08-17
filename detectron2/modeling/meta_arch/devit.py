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
from lib.regionprop import augment_rois, region_coord_2_abs_coord, abs_coord_2_region_coord, SpatialIntegral
from lib.categories import SEEN_CLS_DICT, ALL_CLS_DICT


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
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


def interpolate(seq, T, mode='linear', force=False):
    # seq: B x C x L
    if (seq.shape[-1] < T) or force:
        return F.interpolate(seq, T, mode=mode) 
    else:
    #     # assume is sorted ascending order
        return seq[:, :, -T:]


class PropagateNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, has_input_mask=False, num_layers=3, dropout=0.5, 
                mask_temperature=0.1 # embedding | class
    ):
        super().__init__()
        self.has_input_mask = has_input_mask
        start_mask_dim = 1 if has_input_mask else 0
        self.mask_temperature = mask_temperature

        self.main_layers = nn.ModuleList()
        self.mask_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers

        for i in range(num_layers):
            channels = input_dim if i == 0 else hidden_dim
            self.main_layers.append(nn.Sequential(
                nn.Conv2d(channels + start_mask_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            ))

            self.mask_layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1))
            start_mask_dim += 1
        
        # more proj for regression
        self.class_proj = nn.Linear(hidden_dim, 1)

    def forward(self, embedding, mask=None):
        masks = []
        if self.has_input_mask:
            assert mask is not None 
            masks.append(mask.float())
        
        outputs = []
        for i in range(self.num_layers):
            if len(masks) > 0:
                embedding = torch.cat([embedding,] + masks, dim=1)
            
            embedding = self.main_layers[i](embedding)

            # merge2 (bad)
            # embedding = self.dropout(embedding)

            mask_logits = self.mask_layers[i](embedding) / self.mask_temperature
            mask_weights = mask_logits.sigmoid()
            masks.insert(0, mask_weights) 

            out = {}

            # -------- classification -------- #
            mask_weights = mask_weights / mask_weights.sum(dim=[2, 3], keepdim=True)
            latent = (embedding * mask_weights).sum(dim=[2, 3])
            
            # dropout
            latent = self.dropout(latent)
            
            out['class'] = self.class_proj(latent)
            outputs.append(out)
            

        results = [o['class'] for o in outputs]

        if not self.training:
            results = results[-1]
        return results


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


def distance_embed(x, temperature = 10000, num_pos_feats = 128, scale=10.0):
    # x: [bs, n_dist]
    x = x[..., None]
    scale = 2 * math.pi * scale
    dim_t = torch.arange(num_pos_feats)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    sin_x = x * scale / dim_t.to(x.device)
    emb = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
    return emb # [bs, n_dist, n_emb]


################################################################################################################

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

def elementwise_box_iou(boxes1, boxes2) -> Tuple[torch.Tensor]:
    """Modified from ``torchvision.ops.box_iou``

    Return both intersection-over-union (Jaccard index) and union between
    two sets of boxes.

    Args:
        boxes1: (torch.Tensor[N, 4]): first set of boxes
        boxes2: (torch.Tensor[M, 4]): second set of boxes

    Returns:
        Tuple: A tuple of NxM matrix, with shape `(torch.Tensor[N, M], torch.Tensor[N, M])`,
        containing the pairwise IoU and union values
        for every element in boxes1 and boxes2.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:,1]  # [N,M]

    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union


@META_ARCH_REGISTRY.register()
class OpenSetDetectorWithExamples(nn.Module):

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
                roialign_size=7,
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
                mask_cids = [],
                T_length=128,
                
                bg_cls_weight=0.2,
                batch_size_per_image=128,
                pos_ratio=0.25,
                mult_rpn_score=False,
                num_cls_layers=3,
                use_one_shot= False,
                one_shot_reference= '',
                only_train_mask=True,
                use_mask=True,
                vit_feat_name=None
                ):
        super().__init__()
        if ',' in class_prototypes_file:
            class_prototypes_file = class_prototypes_file.split(',')
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.backbone = backbone # Modify ResNet
        self.bg_cls_weight = bg_cls_weight

        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            # assert input_format == 'RGB'
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
            if 'origin_label_names' in p1 or 'origin_label_names' in p2:
                assert 'origin_label_names' in p2
                oneshot_num_classes = len(p2['origin_label_names'])
                oneshot_sample_pool = len(p2['label_names']) // oneshot_num_classes
                embed_size = p2['prototypes'].shape[-1]
                oneshot_prototypes = p2['prototypes'].reshape(oneshot_num_classes, oneshot_sample_pool, -1, embed_size)

                self.base_prototypes = p1['prototypes']
                self.oneshot_prototypes = oneshot_prototypes

                oneshot_prototypes = oneshot_prototypes[
                    torch.arange(oneshot_num_classes), 
                    torch.randint(0, oneshot_sample_pool, (oneshot_num_classes,))]

                prototypes = torch.cat([p1['prototypes'], oneshot_prototypes], dim=0)
                prototype_label_names = p1['label_names'] + p2['origin_label_names']
            else:
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

        for c in all_cids:
            if c not in prototype_label_names:
                prototype_label_names.append(c)
                mask_cids.append(c)
                class_weights = torch.cat([class_weights, torch.zeros(1, class_weights.shape[-1])], dim=0)
        
        train_class_order = [prototype_label_names.index(c) for c in seen_cids]
        test_class_order = [prototype_label_names.index(c) for c in all_cids]

        self.label_names = prototype_label_names

        assert -1 not in train_class_order and -1 not in test_class_order

        self.register_buffer("train_class_weight", class_weights[torch.as_tensor(train_class_order)])
        self.register_buffer("test_class_weight", class_weights[torch.as_tensor(test_class_order)])
        self.test_class_order = test_class_order

        self.all_labels = all_cids
        self.seen_labels = seen_cids

        self.train_class_mask = None
        self.test_class_mask = None

        if len(mask_cids) > 0:
            self.train_class_mask = torch.as_tensor([c in mask_cids for c in seen_cids])
            if self.train_class_mask.sum().item() == 0:
                self.train_class_mask = None

            self.test_class_mask = torch.as_tensor([c in mask_cids for c in all_cids])

        bg_protos = torch.load(bg_prototypes_file)
        if isinstance(bg_protos, dict):  # NOTE: connect to dict output of `generate_prototypes`
            bg_protos = bg_protos['prototypes']
        if len(bg_protos.shape) == 3:
            bg_protos = bg_protos.flatten(0, 1)
        self.register_buffer("bg_tokens", bg_protos)
        self.num_bg_tokens = len(self.bg_tokens)


        self.roialign_size = roialign_size
        self.roi_align = ROIAlign(roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)
        # input: NCHW, Bx5, output BCKK
        self.box_noise_scale = box_noise_scale


        self.T = T_length
        self.Tpos_emb = 128
        self.Temb = 128
        self.Tbg_emb = 128
        hidden_dim = 256
        # N x C x 14 x 14 -> N x 1 
        self.ce = nn.CrossEntropyLoss()
        self.fc_intra_class = nn.Linear(self.Tpos_emb, self.Temb)
        self.fc_other_class = nn.Linear(self.T, self.Temb)
        self.fc_back_class = nn.Linear(self.num_bg_tokens, self.Tbg_emb)

        cls_input_dim = self.Temb * 2 + self.Tbg_emb
        bg_input_dim = self.Temb + self.Tbg_emb
        
        self.per_cls_cnn = PropagateNet(cls_input_dim, hidden_dim, num_layers=num_cls_layers)
        self.bg_cnn = PropagateNet(bg_input_dim, hidden_dim, num_layers=num_cls_layers)

        self.fc_bg_class = nn.Linear(self.T, self.Temb)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

        self.reg_roialign_size = 20
        self.reg_roi_align = ROIAlign(self.reg_roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)

        reg_feat_dim = self.Temb * 2

        self.rp1 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 1, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp1_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)

        self.rp2 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp2_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)

        self.rp3 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 3, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp3_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)

        self.rp4 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 4, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp4_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)
        
        self.rp5 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 5, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp5_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)

        self.r2c = SpatialIntegral(self.reg_roialign_size)

        self.reg_intra_dist_emb = nn.Linear(self.Tpos_emb, self.Temb)
        self.reg_bg_dist_emb = nn.Linear(self.num_bg_tokens, self.Temb)

        self.cls_temp = cls_temp
        self.evaluation_shortcut = False
        self.num_sample_class = num_sample_class
        self.batch_size_per_image = batch_size_per_image
        self.pos_ratio = pos_ratio
        self.mult_rpn_score = mult_rpn_score

        self.use_one_shot = use_one_shot

        self.one_shot_ref = None

        if use_one_shot:
            self.one_shot_ref = torch.load(one_shot_reference)
        
        # ---------- mask related layers --------- # 
        self.only_train_mask = only_train_mask if use_mask else False
        self.use_mask = use_mask

        self.vit_feat_name = vit_feat_name

        if self.use_mask:

            self.mask_roialign_size = 14
            self.mask_roi_align = ROIAlign(self.mask_roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)

            self.mask_intra_dist_emb = nn.Linear(self.Tpos_emb, self.Temb)
            self.mask_bg_dist_emb = nn.Linear(self.num_bg_tokens, self.Temb)

            num_mask_regression_layers = 5
            self.use_init_mask = True

            layer_start_offset = 1 if self.use_init_mask else 0
            self.use_mask_feat_input = True
            self.use_mask_dropout = True
            self.use_mask_inst_norm = True
            self.use_focal_mask = True
            self.use_mask_ms_feat = True

            feat_inp_dim = 256 if self.use_mask_feat_input else 0

            if self.use_mask_feat_input:
                if self.use_mask_ms_feat:
                    self.mask_feat_compress = nn.ModuleList([nn.Conv2d(self.train_class_weight.shape[-1], feat_inp_dim, 1, 1, 0)
                                                            for _ in range(3)])
                    feat_inp_dim = feat_inp_dim * 3
                else:
                    self.mask_feat_compress = nn.Conv2d(self.train_class_weight.shape[-1], feat_inp_dim, 1, 1, 0)
            
            if self.use_init_mask:
                self.fc_init_mask = nn.Conv2d(1, 1, 1, 1, 0)   
                self.fc_init_mask.weight.data.fill_(1.0)
                self.fc_init_mask.bias.data.fill_(0.0)
            
            if self.use_mask_dropout:
                self.mask_dropout = nn.Dropout2d(p=0.5)
            
            hidden_dim = 384

            self.mp_layers = nn.ModuleList([
                nn.Sequential(
                nn.Conv2d(((self.Temb * 2 + feat_inp_dim) if i == 0 else hidden_dim) + i + layer_start_offset, hidden_dim, 
                        kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(hidden_dim, affine=True) if self.use_mask_inst_norm else nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            ) for i in range(num_mask_regression_layers)])
            self.mp_out_layers = nn.ModuleList([
                nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)
                for i in range(num_mask_regression_layers)
            ])
            self.mask_deconv = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim + num_mask_regression_layers + layer_start_offset, 
                                hidden_dim, kernel_size=2, stride=2, padding=0),
                nn.InstanceNorm2d(hidden_dim, affine=True) if self.use_mask_inst_norm else nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            )
            self.mask_predictor = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)

            if self.only_train_mask:
                self.turn_off_box_training(force=True)
                self.turn_off_cls_training(force=True)
            
    
    def turn_off_cls_training(self, force=False):
        self._turn_off_modules([
            self.fc_intra_class,
            self.fc_other_class,
            self.fc_back_class,
            self.per_cls_cnn,
            self.bg_cnn,
            self.fc_bg_class
        ], force)

    def turn_off_box_training(self, force=False):
        self._turn_off_modules([
            self.reg_intra_dist_emb,
            self.reg_bg_dist_emb,
            self.r2c,
            self.rp1,
            self.rp1_out,
            self.rp2,
            self.rp2_out,
            self.rp3,
            self.rp3_out,
            self.rp4,
            self.rp4_out,
            self.rp5,
            self.rp5_out,
        ], force)

    def _turn_off_modules(self, modules, force):
        for m in modules:
            if m.training or force: 
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

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

        if cfg.DE.OUT_INDICES:
            vit_feat_name = f'res{cfg.DE.OUT_INDICES[-1]}'
        else:
            vit_feat_name = f'res{backbone.n_blocks - 1}'

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
            
            "bg_cls_weight": cfg.DE.BG_CLS_LOSS_WEIGHT,
            "batch_size_per_image": cfg.DE.RCNN_BATCH_SIZE,
            "pos_ratio": cfg.DE.POS_RATIO,
            
            "mult_rpn_score": cfg.DE.MULTIPLY_RPN_SCORE,

            "num_cls_layers": cfg.DE.NUM_CLS_LAYERS,
            
            "use_one_shot": cfg.DE.ONE_SHOT_MODE,
            "one_shot_reference": cfg.DE.ONE_SHOT_REFERENCE,
            
            "only_train_mask": cfg.DE.ONLY_TRAIN_MASK,
            "use_mask": cfg.MODEL.MASK_ON,
            
            "vit_feat_name": vit_feat_name
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
            box_ccwh = box_ccwh + torch.mul(rand_part, diff).cuda() * self.box_noise_scale

            noisy_box = box_cxcywh_to_xyxy(box_ccwh)

            noisy_box[:, 0].clamp_(min=0.0, max=W)
            noisy_box[:, 2].clamp_(min=0.0, max=W)
            noisy_box[:, 1].clamp_(min=0.0, max=H)
            noisy_box[:, 3].clamp_(min=0.0, max=H)

            noisy_boxes.append(noisy_box)

        return noisy_boxes
    
    
    def mask_forward(self, features, boxes, class_labels, class_weights, gt_masks=None, feature_dict=None):
        # all the boxes, labels here are foreground only
        # return pred mask when infernce, or dict of losses when training
        """
        features: B x C x H x W
        class_labels: N
        boxes: N x 4 or N x 5
        class_weights: num_classes x C
        gt_masks: N x polygons
        """
        N = num_masks = len(boxes)
        gt_masks_middle, gt_masks_final = None, None

        assert len(boxes) == len(class_labels)
        if self.training:
            assert gt_masks is not None
            assert len(gt_masks) == len(class_labels)

            boxes4 = boxes
            if boxes4.shape[1] == 5:
                boxes4 = boxes[:, 1:]

            gt_masks_middle = gt_masks.crop_and_resize(boxes4, self.mask_roialign_size).float().to(self.device)
            gt_masks_final = gt_masks.crop_and_resize(boxes4, self.mask_roialign_size * 2).float().to(self.device)

        if boxes.shape[1] == 4:
            boxes = torch.cat([torch.zeros(len(boxes), 1, device=self.device), boxes], dim=1)
        
        roi_feats = self.mask_roi_align(features, boxes)
        roi_feats = roi_feats.flatten(2) # N x C x K2

        bg_roi_feats = roi_feats.transpose(-2, -1) @ self.bg_tokens.T # N x K2 x back
        bg_roi_emb = self.mask_bg_dist_emb(bg_roi_feats)  # N x K2 x T

        fg_roi_feats = roi_feats.transpose(-2, -1) @ class_weights.T # N x K2 x class
        K2 = self.mask_roialign_size ** 2
        fg_roi_feats = torch.gather(fg_roi_feats, 2, class_labels[..., None, None].repeat(1, K2, 1))[:, :, 0]
        fg_roi_emb = distance_embed(fg_roi_feats, num_pos_feats=self.Tpos_emb)
        fg_roi_emb = self.mask_intra_dist_emb(fg_roi_emb)

        bg_roi_emb = bg_roi_emb.permute(0, 2, 1).reshape(N, self.Temb, 
                                                        self.mask_roialign_size, self.mask_roialign_size)
        fg_roi_emb = fg_roi_emb.permute(0, 2, 1).reshape(N, self.Temb, 
                                                        self.mask_roialign_size, self.mask_roialign_size)
        # N x (emb*2) x K x K        
        embedding = torch.cat([fg_roi_emb, bg_roi_emb], dim=1) 
        masks = []
        loss_dict = {}

        if self.use_init_mask:
            init_mask = self.fc_init_mask(fg_roi_feats.reshape(N, 1, self.mask_roialign_size, self.mask_roialign_size))
            if self.training:
                loss_dict[f"mask_bce_loss_0"] = sigmoid_ce_loss(init_mask.flatten(1), 
                                                                gt_masks_middle.flatten(1), num_masks)
                loss_dict[f"mask_dice_loss_0"] = dice_loss(init_mask.flatten(1), 
                                                            gt_masks_middle.flatten(1), num_masks)
            masks.append(init_mask.sigmoid()) 
        
        if self.use_mask_feat_input:
            if self.use_mask_ms_feat:
                assert feature_dict is not None
                ms_feats = [self.mask_roi_align(feature_dict[k], boxes) for k in sorted(feature_dict.keys())] + [roi_feats]
                ms_feats = [m.reshape(N, -1, self.mask_roialign_size, self.mask_roialign_size) for m in ms_feats]
                feat_embs = [self.mask_feat_compress[i](ms_feats[i]) for i in range(3)]    
                embedding = torch.cat([embedding] + feat_embs, dim=1)
            else:
                roi_feats = roi_feats.reshape(N, -1, self.mask_roialign_size, self.mask_roialign_size)
                feat_emb = self.mask_feat_compress(roi_feats)
                embedding = torch.cat([embedding, feat_emb], dim=1)


        mask_temperature = 0.1
        for i, (mp, mp_out) in enumerate(zip(self.mp_layers, self.mp_out_layers)):
            if len(masks) > 0:
                all_mask_tensor = torch.cat(masks, dim=1)
                embedding = torch.cat([embedding, all_mask_tensor], dim=1)
            embedding = mp(embedding)
            if self.use_mask_dropout:
                pred_mask_logits = mp_out(self.mask_dropout(embedding)) / mask_temperature
            else:
                pred_mask_logits = mp_out(embedding) / mask_temperature
            masks.insert(0, pred_mask_logits.sigmoid())

            if self.training:
                loss_dict[f"mask_bce_loss_{len(masks) - 1}"] = sigmoid_ce_loss(pred_mask_logits.flatten(1), 
                                                                gt_masks_middle.flatten(1), num_masks)
                loss_dict[f"mask_dice_loss_{len(masks) - 1}"] = dice_loss(pred_mask_logits.flatten(1), 
                                                            gt_masks_middle.flatten(1), num_masks)
                
                if self.use_focal_mask:
                    loss_dict[f"mask_focal_loss_{len(masks) - 1}"] = sigmoid_focal_loss(
                        pred_mask_logits.flatten(1), gt_masks_middle.flatten(1),
                        alpha=-1, gamma=2.0, reduction="mean")
                
        
        all_mask_tensor = torch.cat(masks, dim=1)
        embedding = torch.cat([embedding, all_mask_tensor], dim=1)
        embedding = self.mask_deconv(embedding)
        if self.use_mask_dropout:
            mask_logits = self.mask_predictor(self.mask_dropout(embedding)) / mask_temperature
        else:
            mask_logits = self.mask_predictor(embedding) / mask_temperature

        if self.training:
            loss_dict[f"mask_bce_loss_out"] = sigmoid_ce_loss(mask_logits.flatten(1), 
                                                                gt_masks_final.flatten(1), num_masks)
            loss_dict[f"mask_dice_loss_out"] = dice_loss(mask_logits.flatten(1), 
                                                            gt_masks_final.flatten(1), num_masks)

            if self.use_focal_mask:
                loss_dict[f"mask_focal_loss_out"] = sigmoid_focal_loss(
                        mask_logits.flatten(1), gt_masks_final.flatten(1),
                        alpha=-1, gamma=2.0, reduction="mean") 
    
        if self.training:
            return loss_dict
        else:
            return mask_logits.sigmoid()


    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        bs = len(batched_inputs)
        loss_dict = {}
        if not self.training: assert bs == 1

        if self.training:
            class_weights = self.train_class_weight
        else:
            if self.use_one_shot:
                # assemble weights on the fly
                class_weights = []
                for c in self.all_labels:
                    if c in self.seen_labels:
                        class_weights.append(self.train_class_weight[self.seen_labels.index(c)].cpu())
                    else:
                        token = random.choice(self.one_shot_ref[c])[1]
                        class_weights.append(token)

                class_weights = F.normalize(torch.stack(class_weights), dim=-1)
                class_weights = class_weights.to(self.device)
            else:
                class_weights = self.test_class_weight

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
        
        with torch.no_grad():
            if self.backbone.training: self.backbone.eval()
            with autocast(enabled=True):
                all_patch_tokens = self.backbone(images.tensor)
                patch_tokens = all_patch_tokens[self.vit_feat_name]
                all_patch_tokens.pop(self.vit_feat_name)
                # patch_tokens = self.backbone(images.tensor)['res11'] 

        if self.training or self.use_one_shot: 
            with torch.no_grad():
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_boxes = [x.gt_boxes.tensor for x in gt_instances]

                rpn_boxes = [x.proposal_boxes.tensor for x in proposals]
                # could try to use only gt_boxes to see the accuracy
                if self.training:
                    noisy_boxes = self.prepare_noisy_boxes(gt_boxes, images.tensor.shape)
                    boxes = [torch.cat([gt_boxes[i], noisy_boxes[i], rpn_boxes[i]]) 
                            for i in range(len(batched_inputs))]
                else:
                    boxes = rpn_boxes

                class_labels = []
                matched_gt_boxes = []
                resampled_proposals = []

                num_bg_samples, num_fg_samples = [], []
                gt_masks = []

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
                        if self.use_mask:
                            gt_masks_i = targets_per_image.gt_masks[matched_idxs[sampled_idxs]]
                    else:
                        gt_boxes_i = torch.zeros(len(sampled_idxs), 4, device=self.device) # not used anyway
                        if self.use_mask:
                            gt_masks_i = PolygonMasks([[np.zeros(6)],] * len(sampled_idxs)).to(self.device)

                    resampled_proposals.append(proposals_per_image)
                    class_labels.append(class_labels_i)
                    matched_gt_boxes.append(gt_boxes_i)
                    if self.use_mask:
                        gt_masks.append(gt_masks_i)

                    num_bg_samples.append((class_labels_i == num_classes).sum().item())
                    num_fg_samples.append(class_labels_i.numel() - num_bg_samples[-1])
                
                if self.training:
                    storage = get_event_storage()
                    storage.put_scalar("fg_count", np.mean(num_fg_samples))
                    storage.put_scalar("bg_count", np.mean(num_bg_samples))

                class_labels = torch.cat(class_labels)
                matched_gt_boxes = torch.cat(matched_gt_boxes) # for regression purpose.
                if self.use_mask:
                    gt_masks = PolygonMasks.cat(gt_masks)
                
                rois = []
                for bid, box in enumerate(resampled_proposals):
                    batch_index = torch.full((len(box), 1), fill_value=float(bid)).to(self.device) 
                    rois.append(torch.cat([batch_index, box], dim=1))
                rois = torch.cat(rois)
        else:
            boxes = proposals[0].proposal_boxes.tensor 
            rois = torch.cat([torch.full((len(boxes), 1), fill_value=0).to(self.device) , 
                            boxes], dim=1)

        roi_features = self.roi_align(patch_tokens, rois) # N, C, k, k
        roi_bs = len(roi_features)

        # roi_features # N x emb x spatial
        #%% #! Classification
        if (self.training and (not self.only_train_mask)) or (not self.training):
            roi_features = roi_features.flatten(2) 
            bs, spatial_size = roi_features.shape[0], roi_features.shape[-1]
            # (N x spatial x emb) @ (emb x class) = N x spatial x class
            feats = roi_features.transpose(-2, -1) @ class_weights.T

            # sample topk classes
            class_topk = self.num_sample_class
            class_indices = None
            if class_topk < 0:
                class_topk = num_classes
                sample_class_enabled = False           
            else:
                if class_topk == 0:
                    class_topk = num_classes
                sample_class_enabled = True

            if sample_class_enabled:
                num_active_classes = class_topk
                init_scores = F.normalize(roi_features.flatten(2).mean(2), dim=1) @ class_weights.T
                topk_class_indices = torch.topk(init_scores, class_topk, dim=1).indices

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
            else:
                num_active_classes = num_classes

            other_classes = [] 
            if sample_class_enabled:
                indexes = torch.arange(0, num_classes, device=self.device)[None, None, :].repeat(bs, spatial_size, 1)
                for i in range(class_topk):
                    cmask = indexes != class_indices[:, i].view(-1, 1, 1)
                    _ = torch.gather(feats, 2, indexes[cmask].view(bs, spatial_size, num_classes - 1)) # N x spatial x classes-1
                    other_classes.append(_[:, :, None, :]) 
            else:
                for c in range(num_classes): # TODO: change to classes sampling during training for LVIS type datasets
                    cmask = torch.ones(num_classes, device=self.device, dtype=torch.bool)
                    cmask[c] = False
                    _ = feats[:, :, cmask] # # N x spatial x classes-1
                    other_classes.append(_[:, :, None, :]) 
            
            other_classes = torch.cat(other_classes, dim=2)  # N x spatial x classes x classes-1
            other_classes = other_classes.permute(0, 2, 1, 3) # N x classes x spatial x classes-1
            other_classes = other_classes.flatten(0, 1) # (Nxclasses) x spatial x classes-1
            other_classes, _ = torch.sort(other_classes, dim=-1)
            other_classes = interpolate(other_classes, self.T, mode='linear') # (Nxclasses) x spatial x T
            other_classes = self.fc_other_class(other_classes) # (Nxclasses) x spatial x emb
            other_classes = other_classes.permute(0, 2, 1) # (Nxclasses) x emb x spatial
            # (Nxclasses) x emb x S x S
            inter_dist_emb = other_classes.reshape(bs * num_active_classes, -1, self.roialign_size, self.roialign_size)

            intra_feats = torch.gather(feats, 2, class_indices[:, None, :].repeat(1, spatial_size, 1)) if sample_class_enabled else feats
            intra_dist_emb = distance_embed(intra_feats.flatten(0, 1), num_pos_feats=self.Tpos_emb) # (Nxspatial) x class x emb TODO: 可以再加一个 linear 层这里
            intra_dist_emb = self.fc_intra_class(intra_dist_emb)
            intra_dist_emb = intra_dist_emb.reshape(bs, spatial_size, num_active_classes, -1)

            # (Nxclasses) x emb x S x S
            intra_dist_emb = intra_dist_emb.permute(0, 2, 3, 1).flatten(0, 1).reshape(bs * num_active_classes, -1, 
                                                                                    self.roialign_size, self.roialign_size)

            bg_feats = roi_features.transpose(-2, -1) @ self.bg_tokens.T # N x spatial x back
            bg_dist_emb = self.fc_back_class(bg_feats) # N x spatial x emb
            bg_dist_emb = bg_dist_emb.permute(0, 2, 1).reshape(bs, -1, self.roialign_size, self.roialign_size)
            # N x emb x S x S

            bg_dist_emb_c = bg_dist_emb[:, None, :, :, :].expand(-1, num_active_classes, -1, -1, -1).flatten(0, 1)
            # (Nxclasses) x emb x S x S

            # (Nxclasses) x EMB x S x S
            per_cls_input = torch.cat([intra_dist_emb, inter_dist_emb, bg_dist_emb_c], dim=1)

            # (Nxclasses) x 1
            cls_logits = self.per_cls_cnn(per_cls_input)

            # N x classes
            if isinstance(cls_logits, list):
                cls_logits = [v.reshape(bs, num_active_classes) for v in cls_logits]
            else:
                cls_logits = cls_logits.reshape(bs, num_active_classes)

            # N x 1
            # feats: N x spatial x class
            cls_dist_feats = interpolate(torch.sort(feats, dim=2).values, self.T, mode='linear') # N x spatial x T
            bg_cls_dist_emb = self.fc_bg_class(cls_dist_feats) # N x spatial x emb
            bg_cls_dist_emb = bg_cls_dist_emb.permute(0, 2, 1).reshape(bs, -1, self.roialign_size, self.roialign_size)
            bg_logits = self.bg_cnn(torch.cat([bg_cls_dist_emb, bg_dist_emb], dim=1))

            if isinstance(bg_logits, list):
                logits = []
                for c,b in zip(cls_logits, bg_logits):
                    logits.append(torch.cat([c, b], dim=1) / self.cls_temp)
            else:
                # N x (classes + 1)
                logits = torch.cat([cls_logits, bg_logits], dim=1)
                logits = logits / self.cls_temp
        else:
            if self.training:
                self.turn_off_cls_training()

        #%% #! Regression
        if (self.training and (not self.only_train_mask)) or (not self.training):
            H,W = images.tensor.shape[2:]
            if self.training:
                fg_indices = class_labels != num_classes 
                matched_gt_boxes = matched_gt_boxes[fg_indices] # nx4
                fg_proposals = rois[fg_indices, 1:] # nx4
                fg_batch_inds = rois[fg_indices, :1] # nx5
                fg_class_labels = class_labels[fg_indices]

                reg_bs = len(fg_proposals)
                aug_rois, pred_roi_mask, gt_roi_mask, covered_flag = augment_rois(fg_proposals, matched_gt_boxes, img_h=H, img_w=W, pooler_size=self.reg_roialign_size, 
                            min_expansion=0.4, expand_shortest=True)
                aug_rois = torch.cat([fg_batch_inds, aug_rois], dim=1)
                gt_region_coords = abs_coord_2_region_coord(aug_rois[:, 1:], matched_gt_boxes, self.reg_roialign_size)

                storage = get_event_storage() 
                storage.put_scalar("roi_cover_ratio", covered_flag.sum().item() / covered_flag.numel())
            else:
                reg_bs = len(rois)
                aug_rois, pred_roi_mask, _, _ = augment_rois(rois[:, 1:], None, img_h=H, img_w=W, pooler_size=self.reg_roialign_size, 
                            min_expansion=0.4, expand_shortest=True)
                aug_rois = torch.cat([rois[:, :1], aug_rois], dim=1)
            
            aroi_feats = self.reg_roi_align(patch_tokens, aug_rois) # N x C x K x K
            aroi_feats = aroi_feats.flatten(2) # N x C x K2
            
            bg_aroi_feats = aroi_feats.transpose(-2, -1) @ self.bg_tokens.T # N x K2 x back
            bg_aroi_emb = self.reg_bg_dist_emb(bg_aroi_feats)  # N x K2 x T

            fg_aroi_feats = aroi_feats.transpose(-2, -1) @ class_weights.T # N x K2 x class
            K2 = self.reg_roialign_size ** 2

            if self.training:
                # N x emb x K x k
                bg_aroi_emb = bg_aroi_emb.permute(0, 2, 1).reshape(reg_bs, self.Temb, self.reg_roialign_size, self.reg_roialign_size)
                # N x K2
                # tmp = torch.zeros(reg_bs, num_classes, device=self.device)
                # tmp[torch.arange(reg_bs, device=self.device), fg_class_labels] = 1.0
                # tmp = tmp[..., None]
                # fg_aroi_feats = torch.bmm(fg_aroi_feats, tmp)[:, :, 0]
                fg_aroi_feats = torch.gather(fg_aroi_feats, 2, fg_class_labels[..., None, None].repeat(1, K2, 1))[:, :, 0]

                fg_aroi_emb = distance_embed(fg_aroi_feats, num_pos_feats=self.Tpos_emb) # N x K2 x emb                   
                fg_aroi_emb = self.reg_intra_dist_emb(fg_aroi_emb)
                # N x emb x K x K
                fg_aroi_emb = fg_aroi_emb.permute(0, 2, 1).reshape(reg_bs, self.Temb, 
                                                                self.reg_roialign_size, self.reg_roialign_size)
                # N x (emb*2) x K x K        
                aroi_emb = torch.cat([fg_aroi_emb, bg_aroi_emb], dim=1) 
            else:
                # (NxK2) x class x emb
                fg_aroi_dist_feats = torch.gather(fg_aroi_feats, 2, class_indices[:, None, :].repeat(1, K2, 1)) if sample_class_enabled else fg_aroi_feats
                fg_aroi_emb = distance_embed(fg_aroi_dist_feats.flatten(0, 1), num_pos_feats=self.Tpos_emb) 
                fg_aroi_emb = self.reg_intra_dist_emb(fg_aroi_emb)
                # N x K2 x class x emb
                fg_aroi_emb = fg_aroi_emb.reshape(reg_bs, K2, num_active_classes, -1)
                # (Nxclass) x emb x K x K
                fg_aroi_emb = fg_aroi_emb.permute(0, 2, 3, 1).flatten(0, 1).reshape(reg_bs * num_active_classes, -1, 
                                                self.reg_roialign_size, self.reg_roialign_size)
                
                bg_aroi_emb = bg_aroi_emb.permute(0, 2, 1).reshape(reg_bs, self.Temb, 
                                    self.reg_roialign_size, self.reg_roialign_size)[:, None, :, :, :].repeat(
                                        1, num_active_classes, 1, 1, 1).flatten(0, 1)
                # (Nxclass) x (emb*2) x K x K
                aroi_emb = torch.cat([fg_aroi_emb, bg_aroi_emb], dim=1) 
                # (Nxclass) x K x K
                pred_roi_mask = pred_roi_mask[:, None, :, :].repeat(1, num_active_classes, 1, 1).flatten(0, 1)
            
            masks = [pred_roi_mask[:, None, :, :].float(), ]
            
            num_masks = len(pred_roi_mask)
            # gt_region_coords 
            embedding = aroi_emb

            if not self.training: 
                aug_rois = aug_rois[:, None, :].repeat(1, num_active_classes, 1).flatten(0, 1)
                

            for i, (rp, rp_out) in enumerate([
                            (self.rp1, self.rp1_out), 
                            (self.rp2, self.rp2_out),
                            (self.rp3, self.rp3_out),
                            (self.rp4, self.rp4_out),
                            (self.rp5, self.rp5_out)]):
                all_mask_tensor = torch.cat(masks, dim=1)
                embedding = torch.cat([embedding, all_mask_tensor], dim=1)
                embedding = rp(embedding)
                pred_mask_logits = rp_out(embedding) / 0.1
                masks.insert(0, pred_mask_logits.sigmoid())

                pred_region_coords = self.r2c(pred_mask_logits)
                if self.training:
                    gt_roi_mask = gt_roi_mask.float()

                    loss_dict[f"aux_bce_loss_{i}"] = sigmoid_ce_loss(pred_mask_logits.flatten(1), gt_roi_mask.flatten(1), num_masks)
                    loss_dict[f"aux_dice_loss_{i}"] = dice_loss(pred_mask_logits.flatten(1), gt_roi_mask.flatten(1), num_masks)

                    # l1, giou
                    loss_dict[f'rg_l1_loss_{i}'] = F.l1_loss(pred_region_coords, gt_region_coords)
                    try:
                        loss_dict[f'rg_giou_loss_{i}'] = (1 - torch.diag(generalized_box_iou(
                                        box_cxcywh_to_xyxy(pred_region_coords),
                                        box_cxcywh_to_xyxy(gt_region_coords)))).mean()
                    except:
                        pass

            # pred_region_coords -> final region coords
            pred_abs_boxes = region_coord_2_abs_coord(aug_rois[:, 1:], pred_region_coords, self.reg_roialign_size)
            fg_pred_deltas = pred_deltas = self.box2box_transform.get_deltas    (
                fg_proposals if self.training else rois[:, None, 1:].repeat(1, num_active_classes, 1).flatten(0, 1), pred_abs_boxes)

            if not self.training:
                pred_deltas = pred_deltas.reshape(reg_bs, num_active_classes, 4)
                pred_deltas = pred_deltas.flatten(1)
        else:
            if self.training:
                self.turn_off_box_training()
        
        #%% #! Loss Finalization and Post Processing
        if self.training:
            class_labels = class_labels.long()
            if not self.only_train_mask:
                if sample_class_enabled:
                    bg_indices = class_labels == num_classes
                    fg_indices = class_labels != num_classes

                    class_labels[fg_indices] = (class_indices == class_labels.view(-1, 1)).nonzero()[:, 1]
                    class_labels[bg_indices] = num_active_classes           

                if isinstance(logits, list):
                    _log_classification_stats(logits[-1].detach(), class_labels)

                    for i, l in enumerate(logits):
                        loss = focal_loss(l, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight)
                        loss_dict[f'focal_loss_{i}'] = loss
                else:
                    _log_classification_stats(logits.detach(), class_labels)
                    loss = focal_loss(logits, class_labels, num_classes=num_active_classes, bg_weight=self.bg_cls_weight)
                    loss_dict['focal_loss'] = loss

            if not self.only_train_mask:
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
                
            if self.use_mask:
                loss_dict.update(
                    self.mask_forward(patch_tokens, rois[fg_indices],
                                        class_labels[fg_indices],
                                        class_weights, gt_masks=gt_masks[fg_indices], feature_dict=all_patch_tokens))
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

            if self.use_one_shot:
                gt_classes = gt_instances[0].gt_classes
                target_class_ids = torch.unique(gt_classes).tolist()
                all_scores = []
                all_boxes = []

                for target_cid in target_class_ids:
                    indices = (class_indices == target_cid).nonzero()
                    roi_inds = indices[:, 0]
                    cls_inds = indices[:, 1]

                    _scores = torch.zeros(len(roi_inds), self.num_classes + 1, device=self.device)
                    _scores[:, target_cid] = scores[roi_inds, cls_inds]
                    _boxes = predict_boxes.reshape(bs, class_topk, 4)[roi_inds, cls_inds]

                    all_scores.append(_scores)
                    all_boxes.append(_boxes)
                
                if len(all_scores) == 0:
                    return []
                else:
                    scores = torch.cat(all_scores)
                    predict_boxes = torch.cat(all_boxes)
            else:
                if sample_class_enabled:
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

            if self.use_mask:
                instances[0].pred_masks = self.mask_forward(patch_tokens, 
                                                        instances[0].pred_boxes.tensor,
                                                        instances[0].pred_classes, 
                                                        class_weights, feature_dict=all_patch_tokens)

            results = self._postprocess(instances, batched_inputs)
            output['instances'] = results[0]['instances']
            return [output, ]