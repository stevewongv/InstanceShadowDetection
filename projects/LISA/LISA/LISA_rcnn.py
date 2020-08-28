#  Copyright (c) Tianyu Wang. All Rights Reserved.
import torch
from torch import nn
from torch.autograd.function import Function

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, RotatedBoxes, pairwise_iou, pairwise_iou_rotated
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference,LightdirectionOutputLayer
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
import numpy as np

# class
@ROI_HEADS_REGISTRY.register()
class LISAROIHeads(StandardROIHeads):
    pass
    """
    This class is used by association RPN.
    """

    # def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
    #     super(associationROIHeads, self).__init__(cfg,input_shape)
    #     pass
        

    def _init_box_head(self,cfg):
        
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )
        self.association_box_head = build_box_head(cfg,pooled_shape)
        self.association_box_predictor = FastRCNNOutputLayers(
                    self.association_box_head.output_size, self.num_classes -1 , cls_agnostic_bbox_reg = False
                )
        self.box_head = build_box_head(cfg,pooled_shape)

        self.box_predictor = FastRCNNOutputLayers(
                    self.box_head.output_size, self.num_classes, cls_agnostic_bbox_reg = False
                )
        self.light_direction_head = build_box_head(cfg,pooled_shape)

        self.light_direction_predictor = LightdirectionOutputLayer(
            self.light_direction_head.output_size)

    def forward(self,images, features, association_proposals, proposals, association_targets=None, targets=None):
        del images
        if self.training:
            association_proposals = self.label_and_sample_proposals(association_proposals, association_targets, True)
            proposals = self.label_and_sample_proposals(proposals,targets)
        del targets
        del association_targets

        features_list = [features[f] for f in self.in_features]
        if self.training:
            losses = self._forward_association_box(features_list,association_proposals)
            losses.update(self._forward_box(features_list, proposals))
            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            losses.update(self._forward_mask(features_list, proposals))
            losses.update(self._forward_keypoint(features_list, association_proposals))
            return proposals, losses
        else:
            
            pred_instances = self._forward_box(features_list, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_associations = self._forward_association_box(features_list,association_proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            
            # pred_associations = self.forward_with_given_association_boxes(features,pred_associations)
            return pred_instances, pred_associations ,{}
    
    
    
    # def forward_with_given_association_boxes(self,feature, instances):

    #     assert not self.training
    #     assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
    #     features = [feature[f] for f in self.in_features]

    #     # instances = self._forward_mask(features, instances)
    #     instances = self._forward_keypoint(features, instances)
    #     return instances



    def _forward_association_box(self, features, association_proposals):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in association_proposals])
        # light_features = self.box_pooler(s)
        light_features = self.light_direction_head(box_features)
        box_features = self.association_box_head(box_features)
        pred_light_direction = self.light_direction_predictor(light_features)
        pred_class_logits, pred_proposal_deltas = self.association_box_predictor(box_features)
        del box_features, light_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            association_proposals,
            self.smooth_l1_beta,
            pred_light_direction
        )
        if self.training:
            return {k+'_asso': v for k, v in outputs.losses().items()}
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            # print(pred_instances)
            return pred_instances
        
    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets, isassociation = False):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if targets[0].has('gt_light'):
            gt_light = [x.gt_light for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(gt_boxes, proposals,gt_light)
        else:
            gt_light = None
            if self.proposal_append_gt:
                proposals = add_ground_truth_to_proposals(gt_boxes,proposals)


        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, proposals_labels = self.proposal_matcher(match_quality_matrix)
            if isassociation:
                num_classes = self.num_classes - 1
            else:
                num_classes = self.num_classes
            # Get the corresponding GT for each proposal
            if has_gt:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # print(gt_classes)
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                
                gt_classes[proposals_labels == 0] = num_classes
                # Label ignore proposals (-1 label)
                gt_classes[proposals_labels == -1] = -1
            else:
                gt_classes = torch.zeros_like(matched_idxs) + num_classes

            sampled_fg_inds, sampled_bg_inds = subsample_labels(
                gt_classes,
                self.batch_size_per_image,
                self.positive_sample_fraction,
                num_classes,
            )

            sampled_inds = torch.cat([sampled_fg_inds, sampled_bg_inds], dim=0)

            proposals_per_image = proposals_per_image[sampled_inds]
            proposals_per_image.gt_classes = gt_classes[sampled_inds]

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_inds]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_inds), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                if gt_light != None:
                    gt_light = Boxes(
                        targets_per_image.gt_light.tensor.new_zeros((len(sampled_inds), 4))
                    )
                    proposal_per_image.gt_light = gt_light

            num_fg_samples.append(sampled_fg_inds.numel())
            num_bg_samples.append(sampled_bg_inds.numel())
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
