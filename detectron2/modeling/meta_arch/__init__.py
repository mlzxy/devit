# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

from .panoptic_fpn import PanopticFPN

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .retinanet import RetinaNet
from .semantic_seg import SEM_SEG_HEADS_REGISTRY, SemanticSegmentor, build_sem_seg_head
from .clip_rcnn import CLIPFastRCNN, PretrainFastRCNN


from .devit import OpenSetDetectorWithExamples
from .devit_update import OpenSetDetectorWithExamples_refactored

__all__ = list(globals().keys())