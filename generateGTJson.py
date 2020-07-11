from datasets import CaptionDataset
from tqdm import tqdm
import os
import pickle
import json
from utils import get_word_synonyms
from nltk import stem
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt

test = CaptionDataset('dataset/output',
                         'coco_5_cap_per_img_5_min_word_freq',
                         'TEST',
                         None,
                         minimal=True)

gtJson = []
imgJson = []

for i, data in tqdm(enumerate(test), total=25000):
    img_id = data[0]
    caption = data[1]
    captionWords = caption[1:-1]
    captionString = ' '.join(captionWords)

    imgDict = {
        'id': img_id
    }
    resultDict = {
        'id': i,
        'image_id': img_id,
        'caption': captionString,
    }
    imgJson.append(imgDict)
    gtJson.append(resultDict)

finalDict = {
    'images': imgJson,
    'annotations': gtJson
}

with open('testGTCaptions.json', 'w') as fp:
    json.dump(finalDict, fp)

exit()