
def matchor(instance, relation):
    def takeTwo(elm):
        return elm[1]
    # count = 0
    results = None
    # for instance,relation  in zip(ins_predictions,rel_predictions):
    # count +=1
    objects = [ i for i,v in enumerate(instance.pred_classes) if v == 1]
    shadows = [i for i,v in enumerate(instance.pred_classes) if v == 0]
    boxes = []
    for o in objects:
        for s in shadows:
            boxes.append(box_combine(o,s,instance.pred_boxes[o].tensor[0].numpy(),instance.pred_boxes[s].tensor[0].numpy()))
    rel_boxes = relation.pred_boxes.tensor.numpy()
    pair = []

    for i,rel_box in enumerate(rel_boxes):
        scores = []
        rel_box = [rel_box[0],rel_box[1],rel_box[2]-rel_box[0],rel_box[3]-rel_box[1]]
        for box in boxes:
            k,v =  box
            
            scores.append([str(i)+'_'+k,compute_iou(rel_box,v)])
        
        if len(rel_boxes) == 1:
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
        ins_relation = ins_predictions.pred_classes * 0
        rel_relation = rel_predictions.pred_classes * 0
        relation_id = 1
        for i in v:
            ins_relation[i[0]] = relation_id
            ins_relation[i[1]] = relation_id
            rel_relation[i[2]] = relation_id
            relation_id += 1
        ins_predictions.pred_relation = ins_relation
        rel_predictions.pred_relation = rel_relation
    return ins_predictions,rel_predictions


if __name__ == "__main__":
    pass