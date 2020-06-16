from nltk import stem
import torch
import numpy as np
from PIL import Image
from textblob import TextBlob
import sys

def separate_objects(img, caption_words, synonyms, annotations, img_id):
    view_results = False
    min_bbox_size = 4
    stemmer1 = stem.snowball.EnglishStemmer()
    stemmer2 = stem.snowball.PorterStemmer()
    blob = TextBlob(' '.join(caption_words))
    classes_to_remove = []
    stemmed_words = [stemmer1.stem(word) for word in caption_words]
    for word in caption_words:
        stemmed_words.append(stemmer2.stem(word))    
    for word in blob.words:
        stemmed_words.append(word.singularize())
    stemmed_words = list(set(stemmed_words))

    annIds = annotations.getAnnIds(img_id)
    anns = annotations.loadAnns(annIds)
    imgInfo = annotations.loadImgs(img_id)
    if len(annIds) == 0 or len(anns) == 0:
        debug("No annotations for file with id {}".format(img_id))
        return None, None
    if len(imgInfo) > 1:
        raise Exception('More than one image for id {}'.format(img_id))
    
    #Logic to match synonyms with more than one word in them
    for word_synonyms_group in synonyms:
        for possible_words in word_synonyms_group:
            should_add = True
            listofwords = list(possible_words.split(' '))
            for individual_word in listofwords:
                if individual_word not in stemmed_words:
                    should_add = False
            if should_add is True:
                classes_to_remove.append(word_synonyms_group[0])

    img_h = imgInfo[0]['height']
    img_w = imgInfo[0]['width']

    img_fg = torch.zeros((3, 256, 256))
    img_bg = img.clone()

    found_objects = False

    for annotation in anns:
        catinfo = annotations.loadCats(annotation['category_id'])[0]
        if catinfo['name'] in classes_to_remove:
            #Transform the values of the bbox to 256x256 dimensions
            bbox = annotation['bbox'].copy()
            bbox[0] = int(bbox[0] / img_w * 256)
            bbox[2] = int(bbox[2] / img_w * 256)
            bbox[1] = int(bbox[1] / img_h * 256)
            bbox[3] = int(bbox[3] / img_h * 256)
            xmin = bbox[0]
            xmax = bbox[0]+bbox[2]
            ymin = bbox[1]
            ymax = bbox[1]+bbox[3]
            xmax = min(xmax, 255)
            ymax = min(ymax, 255)

            if xmin > 255 or xmax > 255 or ymin > 255 or ymax > 255:
                debug("Bounding box out of bounds for {} in img with id {}".format(catinfo['name'], img_id))
                continue

            if bbox[2] > min_bbox_size and bbox[3] > min_bbox_size:
                found_objects = True
                img_fg[:, ymin:ymax, xmin:xmax] = img[:, ymin:ymax, xmin:xmax]
                img_bg[:, ymin:ymax, xmin:xmax] = 0
                
    if found_objects is False:
        debug('No matched foreground objects in image with id {}'.format(img_id))
        return None, None
    if found_objects is True:
        debug("Image with id {} has items that have been removed".format(img_id))
    if view_results is True:
        img_arr = img.permute(1, 2, 0).numpy()
        img_fg_arr = img_fg.permute(1, 2, 0).numpy()
        img_bg_arr = img_bg.permute(1, 2, 0).numpy()
        img = Image.fromarray((img_arr * 255).astype(np.uint8))
        img_fg = Image.fromarray((img_fg_arr * 255).astype(np.uint8))
        img_bg = Image.fromarray((img_bg_arr * 255).astype(np.uint8))
        img.save('img.jpg')
        img_fg.save('img_fg.jpg')
        img_bg.save('img_bg.jpg')
        exit()

    return img_fg, img_bg

def debug(text):
    if '--debug' in sys.argv:
        print('DEBUG - {}'.format(text))
