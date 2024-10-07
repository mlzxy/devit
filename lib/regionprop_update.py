import torch
import numpy as np
import torch.nn as nn
from torchvision.ops import box_iou, box_area
from typing import Tuple
import torch.nn.functional as F


def box_cxcywh_to_xyxy(bbox) -> torch.Tensor:
    cx, cy, w, h = bbox.unbind(-1)
    new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(new_bbox, dim=-1)


def box_xyxy_to_cxcywh(bbox) -> torch.Tensor:
    x0, y0, x1, y1 = bbox.unbind(-1)
    new_bbox = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(new_bbox, dim=-1)

def box_xyxy_clamp(bbox, h, w):
    bbox[:, 0].clamp_(0, w)    
    bbox[:, 2].clamp_(0, w)
    bbox[:, 1].clamp_(0, h)    
    bbox[:, 3].clamp_(0, h)
    return bbox

def box_merge(a, b):
    lti = torch.min(a[:, :2], b[:, :2])
    rbi = torch.max(a[:, 2:], b[:, 2:])
    return torch.cat([lti, rbi], dim=1) # N x 4


def augment_rois(pred_rois, gt_rois=None, img_h=-1, img_w=-1, pooler_size=14, min_expansion=0.3, expand_shortest=True):
    """
    this shall return: 
    1. augmented rois [N x 4]
    2. input and output (target) binary mask [N x K x K]
    3. whether common expansion covers gt [N x 1]
    """
    device = pred_rois.device
    N = len(pred_rois)
    assert img_h > 0 and img_w > 0
    if gt_rois is not None:
        assert len(pred_rois) == len(gt_rois)

    # conventional expand
    tmp = box_xyxy_to_cxcywh(pred_rois)
    if not expand_shortest:
        tmp[:, 2:] *= (1 + min_expansion * 2)
    else:
        expand_amount = tmp[:, 2:] * 2 * min_expansion
        expand_amount = expand_amount.min(dim=1).values
        tmp[:, 2] += expand_amount
        tmp[:, 3] += expand_amount
    expanded_rois = box_cxcywh_to_xyxy(tmp)
    expanded_rois = box_xyxy_clamp(expanded_rois, h=img_h, w=img_w)
    
    # gt guarantee expansion
    if gt_rois is not None:
        gt_expanded_rois = box_merge(pred_rois, gt_rois) # N x 4
        final_rois = box_merge(expanded_rois, gt_expanded_rois)
        covered_flag = torch.all(final_rois == expanded_rois, dim=1)
    else:
        final_rois = expanded_rois
        covered_flag = None

    # create input / output mask
    # first create the coordinate grid, then filter
    x, y = torch.arange(pooler_size, device=device), torch.arange(pooler_size, device=device)
    x, y = x[None, ...].repeat(N, 1), y[None, ...].repeat(N, 1) # N x K
    start_point = final_rois[:, :2] # left top
    wh = box_xyxy_to_cxcywh(final_rois)[:, 2:]
    grid_step = wh / pooler_size
    init_grid_step = grid_step / 2
    x = x * grid_step[:, :1] + init_grid_step[:, :1] + start_point[:, :1] # N x K
    y = y * grid_step[:, 1:] + init_grid_step[:, 1:] + start_point[:, 1:] # N x K

    mask_w = (x >= pred_rois[:, 0].unsqueeze(1)) & (x <= pred_rois[:, 2].unsqueeze(1)) 
    mask_h = (y >= pred_rois[:, 1].unsqueeze(1)) & (y <= pred_rois[:, 3].unsqueeze(1))
    pred_roi_mask = mask_h.reshape(N, -1, 1) * mask_w.reshape(N, 1, -1)

    if gt_rois is not None:
        mask_w = (x >= gt_rois[:, 0].unsqueeze(1)) & (x <= gt_rois[:, 2].unsqueeze(1)) 
        mask_h = (y >= gt_rois[:, 1].unsqueeze(1)) & (y <= gt_rois[:, 3].unsqueeze(1))
        gt_roi_mask = mask_h.reshape(N, -1, 1) * mask_w.reshape(N, 1, -1)
    else:
        gt_roi_mask = None
    return final_rois, pred_roi_mask, gt_roi_mask, covered_flag

# aug_bbox, pred_roi_mask, gt_roi_mask, covered = \
# augment_rois(pred_bbox[None, ...], gt_bbox[None, ...], 
# img_h=image.shape[0], img_w=image.shape[1])


def abs_coord_2_region_coord(regions, boxes, resolution):
    """
    regions: N x 4 in xyxy
    boxes: N x 4 in xyxy
    return: N x 4 in [0, 1] and cxcywh
    """
    wh = box_xyxy_to_cxcywh(regions)[:, 2:]
    init_step = wh / resolution / 2
    wh = wh - init_step
    lt = regions[:, :2] + init_step

    boxes = boxes.clone()
    boxes[:, :2] -= lt
    boxes[:, 2:] -= lt

    boxes = box_xyxy_to_cxcywh(boxes)

    boxes[:, :2] /= wh
    boxes[:, 2:] /= wh
    boxes.clamp_(0, 1)

    return boxes


def region_coord_2_abs_coord(regions, boxes, resolution):
    """
    regions: N x 4 in xyxy
    boxes: N x 4, in [0, 1] and cxcywh
    return: N x 4 in xyxy
    """
    wh = box_xyxy_to_cxcywh(regions)[:, 2:]
    init_step = wh / resolution / 2
    wh = wh - init_step
    lt = regions[:, :2] + init_step
    boxes = boxes.clone()
    boxes[:, :2] *= wh
    boxes[:, 2:] *= wh
    boxes = box_cxcywh_to_xyxy(boxes)
    boxes[:, :2] += lt
    boxes[:, 2:] += lt
    return boxes


def make_region_coord_grid(K):
    x, y = np.linspace(0, 1, K), np.linspace(0, 1, K)
    pos_x, pos_y = np.meshgrid(x, y) # K x K
    pos_x, pos_y = torch.as_tensor(pos_x)[None, ...], torch.as_tensor(pos_y)[None, ...] # 1 x K x K
    return pos_x, pos_y



class SpatialIntegral(nn.Module):

    def __init__(self, resolution, temperature=1.0):
        super().__init__()
        pos_x, pos_y = make_region_coord_grid(resolution)
        self.register_buffer("pos_x", pos_x.flatten(1))
        self.register_buffer("pos_y", pos_y.flatten(1))
        self.temperature = temperature
        
        self.register_parameter("pool_w", nn.Parameter(torch.ones(resolution, 1) / resolution))
        self.register_parameter("pool_h", nn.Parameter(torch.ones(resolution, 1) / resolution))
    
    
    def forward(self, mask_logits):
        # mask_logits: N x 1 x K x K
        bs = len(mask_logits)
        K = mask_logits.shape[-1]
        mask_logits = mask_logits.view(bs, -1) / self.temperature  # N x K2
        softmax_attention = F.softmax(mask_logits, dim=-1)
        c_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)  # N x 1
        c_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)  # N x 1

        sigmoid_activation = mask_logits.sigmoid().reshape(bs, K, K)
        w = (sigmoid_activation.sum(dim=2).sort(dim=1).values / K) @ self.pool_w
        h = (sigmoid_activation.sum(dim=1).sort(dim=1).values / K) @ self.pool_h
        result = torch.cat([c_x, c_y, w, h], dim=1)
        result.clamp_(0, 1)
        return result

Region2Coord = SpatialIntegral



class PropagationLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, in_num_masks, 
                input_hw_size, classification_only=False,
                dropout=0.5, temperature=0.1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim + in_num_masks, out_dim, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Conv2d(out_dim, 1, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(out_dim, 1) 
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = temperature
        self.classification_only = classification_only
        if not classification_only:
            self.region2box = SpatialIntegral(input_hw_size)
    
    def forward(self, x, regions=[]):
        if len(regions) > 0:
            x = torch.cat([x,] + regions, dim=1)
        x = self.conv1(x)
        region_logits = self.conv2(x) / self.temperature
        out_region = region_logits.sigmoid()

        region_weights = out_region / out_region.sum(dim=[2, 3], keepdim=True)
        latent = (x * region_weights).sum(dim=[2, 3])
        latent = self.dropout(latent)
        class_score = self.linear(latent)

        if not self.classification_only:
            box_coords = self.region2box(region_logits)
        else:
            box_coords = None

        return {
            'output_features': x,
            'output_region': out_region,
            'box_coords': box_coords,
            'class_score': class_score
        }


class RegionPropagationNetwork(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, roialign_size, temperature=0.1, dropout=0.5, kernel_size=3, 
                classification_only=False):
        super().__init__()
        self.layers = nn.ModuleList([
            PropagationLayer(
                in_dim=input_dim if i == 0 else hidden_dim, out_dim=hidden_dim, in_num_masks=1+i, 
                input_hw_size=roialign_size, classification_only=classification_only,
                dropout=dropout, temperature=temperature, kernel_size=kernel_size)
            for i in range(num_layers)
        ])
    
    def forward(self, x, init_region):
        regions = [init_region]
        output = []
        keys = ['output_region', 'class_score',  'box_coords'] 
        for layer in self.layers:
            tmp = layer(x, regions)
            x = tmp['output_features']
            regions.append(tmp['output_region'])
            output.append({k: tmp[k] for k in keys})
        return output



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