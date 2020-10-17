#  Copyright (c) Tianyu Wang. All Rights Reserved.
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY, RPN_HEAD_REGISTRY
from detectron2.modeling.proposal_generator.rpn_outputs import RPNOutputs, find_top_rpn_proposals
from detectron2.modeling.proposal_generator.rpn import StandardRPNHead, RPN
from detectron2.structures import BoxMode,Boxes

"""
Registry for LISA RPN heads, which take CNN feature maps and perform
objectness classification and bounding box regression for anchors.
"""

# NOTE: `cfg.MODEL.RPN.HEAD_NAME` should be "LISARPNHead".

def build_rpn_head(cfg, input_shape,shadow_object_part=False):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape,shadow_object_part)


@RPN_HEAD_REGISTRY.register()
class LISARPNHead(StandardRPNHead):
    def __init__(self, cfg, input_shape: List[ShapeSpec], shadow_object_part= False):
        super(LISARPNHead, self).__init__(cfg,input_shape)
        self.shadow_object_part = shadow_object_part
        if self.shadow_object_part:
            in_channels = [s.channels for s in input_shape]
            assert len(set(in_channels)) == 1, "Each level must have the same channel!"
            in_channels = in_channels[0]
            self.conv = nn.Conv2d(in_channels , in_channels, kernel_size=3, stride=1, padding=1)
            for l in [self.conv]:
                nn.init.normal_(l.weight, std=0.01)
                nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        
        pred_objectness_logits = []
        pred_anchor_deltas = []
        if self.shadow_object_part == False:
            pre_features = []
        for i,x in enumerate(features):

            t = F.relu(self.conv(x))
            
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        
        if self.shadow_object_part == False:
            return pred_objectness_logits, pred_anchor_deltas, None
        else:
            return pred_objectness_logits, pred_anchor_deltas


def build_proposal_generator(cfg, input_shape, **args):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.MODEL.PROPOSAL_GENERATOR.NAME
    if name == "PrecomputedProposals":
        return None

    return PROPOSAL_GENERATOR_REGISTRY.get(name)(cfg, input_shape,**args)
    
@PROPOSAL_GENERATOR_REGISTRY.register()
class LISARPN(RPN):

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], shadow_object_part= False):
        super(LISARPN, self).__init__(cfg, input_shape)
        self.shadow_object_part = shadow_object_part
        if self.shadow_object_part:
            self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features], self.shadow_object_part)
    
    def forward(self, images, features, gt_instances=None, pre_proposals=None):
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances

        if self.shadow_object_part == False:
            features = [features[f] for f in self.in_features]
            pred_objectness_logits, pred_anchor_deltas, pre_features = self.rpn_head(features)
            anchors = self.anchor_generator(features)
        else:
            features = [features[f] for f in self.in_features]
            pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
            anchors = self.anchor_generator(features)
            assert len(anchors[0]) == len(pre_proposals), "number of pre_proposals {} and pre_anchors {} should be same.".format(len(anchors[0]),len(pre_proposals))

        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )
        
        if self.training:
            if self.shadow_object_part == False:
                losses = {k+'_asso': v * self.loss_weight for k, v in outputs.losses().items()}
            else:
                losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        with torch.no_grad():

            
            pre_proposals = outputs.predict_proposals()
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                pre_proposals,
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )
            # For RPN-only models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # For end-to-end models, the RPN proposals are an intermediate state
            # and this sorting is actually not needed. But the cost is negligible.
            inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
            proposals = [p[ind] for p, ind in zip(proposals, inds)]
        if self.shadow_object_part == False:
            return proposals, losses, pre_features, pre_proposals
        else:
            return proposals, losses
    


