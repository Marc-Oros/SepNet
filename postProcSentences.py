from datasets import CaptionDataset
from tqdm import tqdm
import os
import pickle
import json
import numpy as np
import spacy
from utils import get_word_synonyms
import re

dataset = CaptionDataset('dataset/output',
                         'coco_5_cap_per_img_5_min_word_freq',
                         'TRAIN',
                         None,
                         minimal=True)

gtJson = []
imgJson = []
nlp = spacy.load("en_core_web_sm")

synonyms = get_word_synonyms()
synonymDict = {}
for item in synonyms:
    synonymDict[item[0]] = item
    

common_pairs = [
    ('person', 'car'),
    ('person', 'chair'),
    ('dining table', 'chair'),
    ('cup', 'bottle'),
    ('person', 'handbag'),
    ('backpack', 'handbag')
]

nouns = [None, None]

modified_sentences = 0
for data in tqdm(dataset):
    #if modified_sentences >= 300:
        #break
    img_id = data[0]
    caption = data[1]
    captionWords = caption[1:-1]
    captionWords = [item for item in captionWords if item != '<unk>']
    captionString = ' '.join(captionWords)
    existsPair = [False, False]
    for pair in common_pairs:
        currSynonyms = [synonymDict[pair[0]], synonymDict[pair[1]]]
        for noun in currSynonyms[0]:
            if re.search(r"\b{}\b".format(noun), captionString) is not None:
                existsPair[0] = True
                nouns[0] = noun
                break
        for noun in currSynonyms[1]:
            if re.search(r"\b{}\b".format(noun), captionString) is not None:
                existsPair[1] = True
                nouns[1] = noun
                break
        if existsPair[0] is True and existsPair[1] is True:
            break
    if existsPair[0] is False or existsPair[1] is False:
        continue
    doc = nlp(captionString)
    wordMask = [1 for i in range(len(doc))]
    for chk in doc.noun_chunks:
        if chk.root.text == nouns[0] or chk.root.text == nouns[1]:
            for word in chk:
                if word.head.text in nouns and word.pos_ not in ['DET', 'PROPN'] and word.text != chk.root.text:
                    wordMask[word.i] = 0
    finalSentence = []
    for word, keep in zip(doc, wordMask):
        if keep == 1:
            finalSentence.append(word.text)
    finalSentence = ' '.join(finalSentence)
    if len(doc) != sum(wordMask):
        modified_sentences += 1
        print(nouns)
        print(captionString)
        print(finalSentence)

exit()
