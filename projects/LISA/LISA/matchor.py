
def matchor(instance, association):
    def takeTwo(elm):
        return elm[1]
    # count = 0
    results = None
    # for instance,association  in zip(ins_predictions,ass_predictions):
    # count +=1
    objects = [ i for i,v in enumerate(instance.pred_classes) if v == 1]
    shadows = [i for i,v in enumerate(instance.pred_classes) if v == 0]
    boxes = []
    for o in objects:
        for s in shadows:
            boxes.append(box_combine(o,s,instance.pred_boxes[o].tensor[0].numpy(),instance.pred_boxes[s].tensor[0].numpy()))
    ass_boxes = association.pred_boxes.tensor.numpy()
    pair = []

    for i,ass_box in enumerate(ass_boxes):
        scores = []
        ass_box = [ass_box[0],ass_box[1],ass_box[2]-ass_box[0],ass_box[3]-ass_box[1]]
        for box in boxes:
            k,v =  box
            
            scores.append([str(i)+'_'+k,compute_iou(ass_box,v)])
        
        if len(ass_boxes) == 1:
            pair.append(sorted(scores,key=takeTwo,reverse=True)[:1])
        else:
            pair.append(sorted(scores,key=takeTwo,reverse=True)[:1])
            if not sum([sc[1] > 0.7 for sc in pair[i]]):
                pair[i] = [[0,0]]
    O = {}
    S = {}
    for k,v in enumerate(pair):
        if v != [[0,0]] and v != []:
            r,o,s = v[0][0].split('_')
            if o in O or s in S:
                if o in O:
                    if v[0][1] > O[o][1]:
                        O[o] = v[0]
                if s in S:
                    if v[0][1] > S[s][1]:
                        S[s] = s[0]
            else:
                O[o] = v[0]
                S[s] = v[0]
    for k,v in S.items():
        try:
            r,o,s = v[0].split('_')
            if results == None:
                results = [[int(o),int(s),int(r)]]
            else:
                results.append([int(o),int(s),int(r)])
        except:
            return None,None
            
    for v in results:
        ins_association = ins_predictions.pred_classes * 0
        ass_association = ass_predictions.pred_classes * 0
        association_id = 1
        for i in v:
            ins_association[i[0]] = association_id
            ins_association[i[1]] = association_id
            ass_association[i[2]] = association_id
            association_id += 1
        ins_predictions.pred_association = ins_association
        ass_predictions.pred_association = ass_association
    return ins_predictions,ass_predictions


if __name__ == "__main__":
    pass
