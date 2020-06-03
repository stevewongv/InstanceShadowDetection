#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager
from LISA.matchor import matchor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from detectron2.data.datasets import register_soba_instances
register_soba_instances("soba_cast_shadow_train_full", {}, "../../../SOBA/annotations/SOBA_train.json", "../..././SOBA/SOBA/")
register_soba_instances("soba_cast_shadow_val_full",{},"../../../SOBA/annotations/SOBA_val.json","../../../SOBA/SOBA/")



def create_instances(ins_predictions, ass_predictions,image_size):
    ret = Instances(image_size)

    ins_score = np.asarray([x["score"] for x in ins_predictions])
    ins_association = np.asarray([x["association_id"] for x in ins_predictions])
    chosen = (ins_score > args.conf_threshold).nonzero()[0]
    ins_score = ins_score[chosen]
    ins_association = ins_association[chosen]
    bbox = np.asarray([ins_predictions[i]["bbox"] for i in chosen])
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(ins_predictions[i]["category_id"]) for i in chosen])
    ret.scores = ins_score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels
    ret.pred_associations = ins_association

    try:
        ret.pred_masks = [ins_predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    ass_ret = Instances(image_size)
    ass_score = np.asarray([x["score"] for x in ass_predictions])
    ass_ass = np.asarray([x["association_id"] for x in ass_predictions])
    chosen = (ass_score > args.conf_threshold).nonzero()[0]
    ass_score = ass_score[chosen]
    ass_ass = ass_ass[chosen]
    bbox = np.asarray([ass_predictions[i]["bbox"] for i in chosen])
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    
    labels = np.asarray([dataset_ass_id_map(ass_predictions[i]["category_id"]) for i in chosen])

    ass_ret.scores = ass_score
    ass_ret.pred_boxes = Boxes(bbox)
    ass_ret.pred_classes = labels
    ass_ret.pred_associations = ass_ass


    return ret,ass_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--ins_input", required=True, help="JSON file produced by the model")
    parser.add_argument("--ass_input", required=True)
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()
    with PathManager.open(args.ins_input, "r") as f:
        ins_predictions = json.load(f)
    with PathManager.open(args.ass_input,'r') as f:
        ass_predictions = json.load(f)

    ins_pred_by_image = defaultdict(list)
    ass_pred_by_image = defaultdict(list)
    
    for p in ins_predictions:
        ins_pred_by_image[p["image_id"]].append(p)
    for p in ass_predictions:
        ass_pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]
        def dataset_ass_id_map(ds_id):
            return metadata.association_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])
        ins_predictions,ass_predictions = create_instances(ins_pred_by_image[dic["image_id"]], ass_pred_by_image[dic["image_id"]],img.shape[:2])
        # ins_predictions,ass_predictions = matchor(ins_predictions,ass_predictions)
        if ins_predictions == None:
            continue
        vis = Visualizer(img, metadata)
        # vis = Visualizer(img,metadata)

        vis_assa = vis.draw_instance_predictions(ass_predictions,True,labels_align='right').get_image()
        # cv2.imwrite(os.path.join(args.output, 'ass_'+basename), vis_assa[:,:,::-1])
        vis_pred = vis.draw_instance_predictions(ins_predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()
        h,_,_ = img.shape

        white = np.ones((h,20,3),dtype=('uint8'))*255

        concat = np.concatenate((img,white,vis_gt,white,white,vis_pred), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])

