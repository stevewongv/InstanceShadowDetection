import numpy as np
import pysobatools.sobaeval as SOAPEval
import pysobatools.cocoeval as Eval
from pysobatools.soba import SOBA
import json


soba = SOBA('../../../SOBA/annotations/SOBA_val.json')
results =  json.load(open('./output_light/inference/soba_instances_results.json'))
association = json.load(open('./output_light/inference/soba_association_results.json'))

instance_soba = soba.loadRes(results)
association_soba = soba.loadRes_asso(association)

sobaeval= SOAPEval.SOAPeval(soba,instance_soba,association_soba)
print('segmentaion:')

sobaeval.evaluate_asso()

sobaeval.accumulate()
sobaeval.summarize()
print('bbox:')
sobaeval= SOAPEval.SOAPeval(soba,instance_soba,association_soba)
sobaeval.params.iouType = 'bbox'
sobaeval.evaluate_asso()

sobaeval.accumulate()
sobaeval.summarize()

print("--------------")
sobaeval= Eval.COCOeval(soba,association_soba)
sobaeval.evaluate_asso()
sobaeval.accumulate()
sobaeval.summarize()

sobaeval= Eval.COCOeval(soba,association_soba)
sobaeval.params.iouType="bbox"
sobaeval.evaluate_asso()
sobaeval.accumulate()
sobaeval.summarize()