#  Copyright (c) Tianyu Wang. All Rights Reserved.
import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess, matchor, combine_association
from .LISA_rpn  import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import __all__, GeneralizedRCNN
from detectron2.utils.registry import Registry
__all__.append("LISARCNN")

@META_ARCH_REGISTRY.register()
class LISARCNN(GeneralizedRCNN):

    def __init__(self,cfg):
        super(LISARCNN, self).__init__(cfg)
        self.association_proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape(), shadow_object_part= False)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape(), shadow_object_part= True)
        self.to(self.device)
    def forward(self, batched_inputs):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        if "associations" in batched_inputs[0]:
            gt_associations = [x["associations"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.association_proposal_generator:
            association_proposals, association_losses, pre_features, pre_proposals = self.association_proposal_generator(images, features, gt_associations)
        
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images,features,gt_instances,pre_proposals)

        _, detector_losses = self.roi_heads(images, features, association_proposals, proposals, gt_associations, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(association_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.association_proposal_generator:
                association_proposals, _, pre_features, pre_proposals = self.association_proposal_generator(images, features)
            else:
                assert "associations" in batched_inputs[0]
                proposals = [x["associations"].to(self.device) for x in batched_inputs]
            if self.proposal_generator:
                # concat_features = {}
                # for pre_features,(k,v) in zip(pre_features,features.items()):
                #     concat_features[k] = torch.cat([v,pre_features],1)
                proposals, _ = self.proposal_generator(images,features,pre_proposals = pre_proposals)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results,associations, _ = self.roi_heads(images, features, association_proposals, proposals, None, None)
        
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r.to(torch.device('cpu'))})

            processed_associations = []
            for results_per_image, input_per_image, image_size in zip(
                associations, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_associations.append({"instances": r.to(torch.device('cpu'))})
            
            for instances, associations in zip(processed_results, processed_associations):
                _instances, _associations = matchor(instances["instances"],associations["instances"])
                _associations,_instances = combine_association(_instances,_associations)
                associations["instances"] = _associations
                instances["instances"] = _instances
                    

            return processed_results,processed_associations
        else:
            return results,associations




