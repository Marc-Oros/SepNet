from datasets import *
from tqdm import tqdm
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import os
import pickle
from utils import getId2ClassMap
import json

#Detectron2 imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

#Detectron2 setup
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

# Create predictor
predictor = DefaultPredictor(cfg)

data_folder = 'dataset/output'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

train_annotations=COCO(os.path.join('dataset', 'annotations', 'instances_train2014.json'))
val_annotations=COCO(os.path.join('dataset', 'annotations', 'instances_val2014.json'))

clsMap = getId2ClassMap()

gt = json.load(open('dataset/annotations/instances_train2014.json', 'r'))

cocoDict = {}
cocoDict['info'] = gt['info']
cocoDict['licenses'] = gt['licenses']
cocoDict['images'] = []
cocoDict['annotations'] = []
cocoDict['categories'] = gt['categories']

classMapping = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
rev_cls_map = {v: k for k, v in classMapping.items()}

dataset = FasterRCNNDataset(data_folder, data_name, 'TEST', transform=None)
preds = {}
annId = 0
for i, (img_id, img, img_h, img_w) in tqdm(enumerate(dataset), total=len(dataset)):
    if i % 5 != 0:
        continue

    outputs = predictor(255 * img.permute(1, 2, 0).numpy())
    
    imgInfo = {
        "license": 4,
        "file_name": "",
        "coco_url": "",
        "height": img_h,
        "width": img_w,
        "date_captured": "",
        "flickr_url": "",
        "id": img_id
    }

    for bbox, clsId in zip(outputs['instances'].pred_boxes, outputs['instances'].pred_classes):
        bbox = bbox.cpu().numpy()
        bbox_orig = bbox.copy()
        # Upscale bboxes to img size
        bbox[0] = bbox[0] * img_w / 256
        bbox[2] = bbox[2] * img_w / 256
        bbox[1] = bbox[1] * img_h / 256
        bbox[3] = bbox[3] * img_h / 256
        # Convert bboxes to XYWH format, as coco annotations
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        
        cocoDict['annotations'].append({
            "segmentation": [],
            "area": 0,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": bbox.tolist(),
            "category_id": rev_cls_map[clsId.item()],
            "id": annId
        })
        annId += 1
    cocoDict['images'].append(imgInfo)
with (open('coco_FRCNN.json', 'w')) as f:
    json.dump(cocoDict, f)

    