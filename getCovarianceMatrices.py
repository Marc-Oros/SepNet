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

train = CaptionDataset('dataset/output',
                         'coco_5_cap_per_img_5_min_word_freq',
                         'TRAIN',
                         None,
                         minimal=True)

train_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_train2014.json'))
val_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_val2014.json'))
idx2dataset = {}
with(open(os.path.join('dataset', 'output', 'TRAIN_ids.txt'), 'r')) as f:
    for i, line in enumerate(f):
        values = line.rstrip().split()
        idx2dataset[i] = values[1]

synonyms = get_word_synonyms()
stemmer = stem.snowball.PorterStemmer()

id2class = []
for id, word_synonyms_group in enumerate(synonyms):
    id2class.append(word_synonyms_group[0])

class2Id = {}
for i, item in enumerate(id2class):
    class2Id[item] = i


covMatrixCaps = np.zeros((80, 80), dtype=np.int)
covMatrixAnns = np.zeros((80, 80), dtype=np.int)

for img_id, caption in tqdm(train):
    captionWords = caption[1:-1]
    
    #Covariance matrix from captions

    stemmed_words = [stemmer.stem(word) for word in captionWords]
    stemmedCaption = ' '.join(stemmed_words)
    classes_in_caption = []

    #Logic to match synonyms with more than one word in them
    for id, word_synonyms_group in enumerate(synonyms):
        for possible_synonym in word_synonyms_group:
            if possible_synonym in stemmedCaption:
                classes_in_caption.append(id)
                
    classes_in_caption = set(classes_in_caption)

    for firstClass in classes_in_caption:
        for secondClass in classes_in_caption:
            if firstClass == secondClass:
                continue
            covMatrixCaps[firstClass, secondClass] += 1


    #Covariance matrix from annotations
    classes_in_caption = []
    annotations = train_annotations if idx2dataset[i // 5] == 'train' else val_annotations
    annIds = annotations.getAnnIds(img_id)
    imgAnns = annotations.loadAnns(annIds)

    for annotation in imgAnns:
        catinfo = annotations.loadCats(annotation['category_id'])[0]
        categoryName = catinfo['name']
        classes_in_caption.append(class2Id[categoryName])
    classes_in_caption = set(classes_in_caption)

    for firstClass in classes_in_caption:
        for secondClass in classes_in_caption:
            if firstClass == secondClass:
                continue
            covMatrixAnns[firstClass, secondClass] += 1


#Plot covariance matrix from captions
fig, ax = plt.subplots(figsize=(20,10))

ax.matshow(covMatrixCaps, cmap='hot')
ax.set_title('Covariance matrix from captions', pad=50)
ax.set(xticks=np.arange(len(id2class)), xticklabels=id2class,
       yticks=np.arange(len(id2class)), yticklabels=id2class)
plt.xticks(rotation=90)
plt.savefig('covMatrixCaps.png')

#Plot covariance matrix from annotations
ax.matshow(covMatrixAnns, cmap='hot')
ax.set_title('Covariance matrix from annotations', pad=50)
ax.set(xticks=np.arange(len(id2class)), xticklabels=id2class,
       yticks=np.arange(len(id2class)), yticklabels=id2class)
plt.xticks(rotation=90)
plt.savefig('covMatrixAnns.png')

#Dump results to files
pickle.dump(covMatrixCaps, open("covMatrixCaps.pkl", "wb"))
pickle.dump(covMatrixAnns, open("covMatrixAnns.pkl", "wb"))
pickle.dump(id2class, open("idxToClass.pkl", "wb"))

np.savetxt("covMatrixCaps.csv", covMatrixCaps, delimiter=",")
np.savetxt("covMatrixAnns.csv", covMatrixAnns, delimiter=",")

exit()