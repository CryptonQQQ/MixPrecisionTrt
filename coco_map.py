# -*- coding:utf-8 -*-
import json
import pylab
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

pylab.rcParams['figure.figsize'] = (10.0, 8.0)


def get_img_id(file_name):
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset


def calculate_coco_mAP():
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]
    cocoGt_file = './instances_val2014.json'
    cocoGt = COCO(cocoGt_file)
    cocoDt_file = 'result/mix_res.json'
    imgIds = get_img_id(cocoDt_file)
    print(len(imgIds))
    cocoDt = cocoGt.loadRes(cocoDt_file)
    imgIds = sorted(imgIds)
    imgIds = imgIds[0:5000]
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    calculate_coco_mAP()
