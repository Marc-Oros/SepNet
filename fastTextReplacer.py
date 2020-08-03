import pickle
import os
from pycocotools.coco import COCO
import torch
from image_manipulation import debug

class FastTextReplacer:
    def __init__(self, train_annotations=None, val_annotations=None, useFRCNN=False):
        self.useFRCNN = useFRCNN
        if useFRCNN is False:
            if train_annotations is None:
                self.train_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_train2014.json'))
            else:
                self.train_annotations = train_annotations
            if val_annotations is None:
                self.val_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_val2014.json'))
            else:
                self.val_annotations = val_annotations
        else:
            self.test_annotations = COCO('coco_FRCNN.json')
        self.classMap = pickle.load(open('classVectors.pkl', 'rb'))

    def replace(self, img_id, tensor, pair):
        #Image tensors as (height, width)
        if self.useFRCNN is False:
            annotations = self.train_annotations
            annIds = annotations.getAnnIds(img_id)
            if len(annIds) == 0:
                annotations = self.val_annotations
                annIds = annotations.getAnnIds(img_id)
                if len(annIds) == 0:
                    raise Exception('Image ID {} without annotations'.format(img_id))
        else:
            annotations = self.test_annotations
            annIds = annotations.getAnnIds(img_id)
        anns = annotations.loadAnns(annIds)

        imgInfo = annotations.loadImgs(img_id)
        tensor_h = tensor.shape[0]
        tensor_w = tensor.shape[1]
        img_h = imgInfo[0]['height']
        img_w = imgInfo[0]['width']

        for annotation in anns:
            catinfo = annotations.loadCats(annotation['category_id'])[0]
            if pair[0] == catinfo['name']:
                wordVec = torch.Tensor(self.classMap[pair[1]])
            else:
                wordVec = torch.Tensor(self.classMap[catinfo['name']])
            #Transform the values of the bbox to the correct dimensions
            bbox = annotation['bbox'].copy()
            bbox[0] = int(bbox[0] / img_w * tensor_w)
            bbox[2] = int(bbox[2] / img_w * tensor_w)
            bbox[1] = int(bbox[1] / img_h * tensor_h)
            bbox[3] = int(bbox[3] / img_h * tensor_h)
            xmin = bbox[0]
            xmax = bbox[0]+bbox[2]
            ymin = bbox[1]
            ymax = bbox[1]+bbox[3]
            xmax = min(xmax, tensor_w)
            ymax = min(ymax, tensor_h)

            if xmin > tensor_w or xmax > tensor_w or ymin > tensor_h or ymax > tensor_h:
                debug("Bounding box out of bounds for {} in img with id {}".format(catinfo['name'], img_id))
                continue

            possibleY = range(ymin, ymax)
            possibleX = range(xmin, xmax)
            if len(possibleX) == 0:
                possibleX = [xmin]
            if len(possibleY) == 0:
                possibleY = [ymin]
            for i in possibleY:
                for j in possibleX:
                    tensor[i, j, :] = wordVec
        return tensor