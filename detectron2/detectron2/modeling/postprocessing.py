# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn import functional as F
import math
import numpy as np

from detectron2.layers import paste_masks_in_image
from detectron2.structures import Boxes, BoxMode, Instances
import pycocotools.cocoeval as eval


def decode(segm):
    return eval.maskUtils.decode(segm).astype('uint8')


def encode(segm):
    return eval.maskUtils.encode(segm)


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.tensor[:, 0::2] *= scale_x
    output_boxes.tensor[:, 1::2] *= scale_y
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = paste_masks_in_image(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result

def takeTwo(elm):
    return elm[1]
#
#     """
#     param:
#     rec1: (x0, y0, w, h)
#     rec2: (x0, y0, w, h)
#     x0, y0: the upper left point of rec.
#     w, h:  the length and width of rec.
#     """


def compute_iou(rec1, rec2):
    left_x = max(rec1[0], rec2[0])
    left_y = max(rec1[1], rec2[1])
    right_x = min(rec1[0] + rec1[2], rec2[0] + rec2[2])
    right_y = min(rec1[1] + rec1[3], rec2[1] + rec2[3])
    if left_x >= right_x or left_y >= right_y:
        return 0
    else:
        S_mid = (right_y - left_y) * (right_x - left_x)
        S_total = (rec1[2] * rec1[3]) + (rec2[2] * rec2[3]) - S_mid
        return S_mid / S_total


def box_combine(o, s, box1, box2):
    """
    args:
        box1 : (x1_0,  y1_0,  x1_1, y1_1)
        box2: (x2_0, y2_0, x2_1, y2_1)
    return:
        dict["1_2":(min(x1_0,x2_0),min(y1_0,y2_0),max(x1_1,x2-1),max(y2_1,y2_2))]
    """
    name = '{}_{}'.format(o, s)
    combine = (min(box1[0], box2[0]), min(box1[1], box2[1]),
               max(box1[2], box2[2]), max(box1[3], box2[3]))
    combine = (combine[0], combine[1], combine[2] - combine[0],
               combine[3] - combine[1])  # XYXY to XYWH
    return [name, combine]


def compute_direction(box1,box2):
    pass

def rect_distance(a, b):
    x1, y1, x1b, y1b = a
    x2, y2, x2b, y2b = b
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:
        return 0


def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def matchor(instance, association):
    results = []
    objects = [i for i, v in enumerate(instance.pred_classes) if v == 0]
    shadows = [i for i, v in enumerate(instance.pred_classes) if v == 1]
    boxes = []
    for o in objects:
        if instance.scores[o] < 0.5:
            continue
        for s in shadows:
            if instance.scores[s] < 0.5:
                continue
            o_box = instance.pred_boxes[o].tensor[0].numpy()
            s_box = instance.pred_boxes[s].tensor[0].numpy()
            o_area = (o_box[2] - o_box[0]) * (o_box[3] - o_box[1])
            s_area = (s_box[2] - s_box[0]) * (s_box[3] - s_box[1])
            if compute_iou((o_box[0], o_box[1], o_box[2] - o_box[0], o_box[3] - o_box[1]), (s_box[0], s_box[1], s_box[2] - s_box[0], s_box[3] - s_box[1])) == 0:
                if rect_distance(o_box, s_box) >= s_box[3] -  s_box[1]:
                    continue
            boxes.append(box_combine(o, s, o_box, s_box))
    ass_boxes = association.pred_boxes.tensor.numpy()
    pair = []

    for i, ass_box in enumerate(ass_boxes):
        scores = []
        ass_box = [ass_box[0], ass_box[1], ass_box[2] - ass_box[0], ass_box[3] - ass_box[1]]
        for box in boxes:
            k, v = box

            scores.append([str(i) + '_' + k, compute_iou(ass_box, v)])

        if len(ass_boxes) == 1:
            pair.append(sorted(scores, key=takeTwo, reverse=True)[:1])
        else:
            pair.append(sorted(scores, key=takeTwo, reverse=True)[:1])
            if not sum([sc[1] > 0.5 for sc in pair[i]]):
                pair[i] = [[0, 0]]
    O = {}
    S = {}
    for k, v in enumerate(pair):
        if v != [[0, 0]] and v != []:
            r, o, s = v[0][0].split('_')
            if o in O:
                if s in S:
                    if v[0][1] > O[o][1] and v[0][1] > S[s][1]:
                        O[o] = v[0]
                        S[s] = v[0]
                else:
                    if v[0][1] > O[o][1]:
                        O[o] = v[0]
            elif s in S:
                if v[0][1] > S[s][1]:
                    S[s] = v[0]                    
            else:
                O[o] = v[0]
                S[s] = v[0]
    for k, v in S.items():
        r, o, s = v[0].split('_')
        results.append((int(o), int(s), int(r)))

    ins_association = instance.pred_classes * 0
    ret_association = association.pred_classes * 0
    if results == []:
        instance.pred_associations = ins_association
        association.pred_associations = ret_association
        return instance, association
    association_id = 1

    for i in results:
        if ins_association[i[0]]+ins_association[i[1]] == 0:
            ins_association[i[0]] = association_id
            ins_association[i[1]] = association_id
            ret_association[i[2]] = association_id
            association_id += 1

    instance.pred_associations = ins_association
    association.pred_associations = ret_association
    return instance, association


def combine_association(instance, association):
    pred_masks = [mask.numpy() for mask in instance.pred_masks]
    pred_scores = instance.scores.numpy()
    pred_boxes = instance.pred_boxes.tensor.numpy().tolist()
    pred_classes = instance.pred_classes.numpy()
    h, w = pred_masks[0].shape
    pred_associations = instance.pred_associations.numpy()
    pred_light = association.pred_light.tensor.numpy()
    ret = Instances((h,w))
    ins = Instances((h,w))

    if np.sum(pred_associations) == 0:
        ret.pred_boxes = association.pred_boxes
        ret.scores = association.scores
        ret.pred_classes = association.pred_classes
        ret.pred_light = association.pred_light.tensor.numpy().tolist()
        segm = np.zeros((h,w,1),order='F',dtype='uint8')
        ret.pred_masks = [segm] * len(association.pred_boxes)
        ret.pred_associations = association.pred_associations.numpy().astype('int').tolist()
        instance.pred_associations = pred_associations.astype('int').tolist()
        return ret,instance

    mask_map = {}
    for i, ass in enumerate(pred_associations):
        if ass != 0:
            if ass in mask_map:
                if pred_classes[i] == 1:
                    mask_map[ass].append((pred_masks[i], pred_scores[i],pred_classes[i],pred_boxes[i]))
                else:
                    mask_map[ass] = [(pred_masks[i], pred_scores[i],pred_classes[i],pred_boxes[i]),mask_map[ass][0]]
            else:
                
                mask_map[ass] = [(pred_masks[i], pred_scores[i],pred_classes[i],pred_boxes[i])]

    results = []
    boxes = []
    scores = []
    classes = []
    associations = []
    light = []

    for i,ass in enumerate(association.pred_associations):
        if ass != 0:
            light.append(pred_light[i].tolist())

    for k, v in mask_map.items():
        associations.append(int(k))
        s, o = v
        avg_score = float((s[1]+ o[1])/2)
        _s = s[0].reshape(h,w,1)
        _o = o[0].reshape(h,w,1)

        comb = _s + _o
        classes.append(0)
        segm = encode(np.array(comb,order='F',dtype='uint8'))[0]
        boxes.append(BoxMode.convert(eval.maskUtils.toBbox(segm), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS))
        results.append(comb)
        scores.append(avg_score)

    ret.pred_masks = results
    ret.pred_boxes = boxes
    ret.scores = scores
    ret.pred_classes = classes
    ret.pred_associations = associations
    ret.pred_light= light

    instance.pred_associations = instance.pred_associations.numpy().astype('int').tolist()

    return ret,instance
