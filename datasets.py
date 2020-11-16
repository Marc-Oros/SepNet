import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from image_manipulation import separate_objects
from utils import get_word_synonyms, getClsList, getId2ClassMap, getClass2IdMap, updateCovMatrix, getClassCombinationsWithPerson
from pycocotools.coco import COCO
import spacy
import re
from sentenceSimplifier import SentenceSimplifier
import random
from fastTextReplacer import FastTextReplacer
import numpy as np
import pickle
import itertools
import sys


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None, minimal=False):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        self.minimal = minimal

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # Load word map (word2ix)
        with open('dataset/output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json', 'r') as j:
            word_map = json.load(j)
        self.word_map = {v: k for k, v in word_map.items()}  # ix2word

        self.idx2id = {}
        with(open(os.path.join(data_folder, self.split + '_ids.txt'), 'r')) as f:
            for i, line in enumerate(f):
                values = line.rstrip().split()
                self.idx2id[i] = int(values[0])

    def __getitem__(self, i):
        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        caption_words = [self.word_map[caption[idx].item()] for idx in range(caplen)]

        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        if self.minimal is True:
            return self.idx2id[i // self.cpi], caption_words, img

        if self.transform is not None:
            img = self.transform(img)

        if self.split is 'TRAIN':
            return self.idx2id[i // self.cpi], img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return self.idx2id[i // self.cpi], img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

    def getImgId(self, i):
        return self.idx2id[i // self.cpi]


class CaptionDatasetSplit(CaptionDataset):
    def __init__(self, data_folder, data_name, split, transform=None, train_annotations=None, val_annotations=None):
        super().__init__(data_folder, data_name, split, None, False)
        
        self.usableImages = 0
        self.splitTransform = transform
        self.synonyms = get_word_synonyms()
        if train_annotations is None:
            self.train_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_train2014.json'))
        else:
            self.train_annotations = train_annotations
        if val_annotations is None:
            self.val_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_val2014.json'))
        else:
            self.val_annotations = val_annotations
        if self.split == 'TEST' and '-frcnn' in sys.argv:
            print('Using Faster R-CNN annotations')
            self.test_annotations = COCO('coco_FRCNN.json')
        self.idx2dataset = {}
        
        with(open(os.path.join(data_folder, self.split + '_ids.txt'), 'r')) as f:
            for i, line in enumerate(f):
                values = line.rstrip().split()
                self.idx2dataset[i] = values[1]

    def __getitem__(self, i):
        if self.split is 'TRAIN':
            img_id, img, caption, caplen = super().__getitem__(i)
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            img_id, img, caption, caplen, all_captions = super().__getitem__(i)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        caption_words = [self.word_map[caption[idx].item()] for idx in range(caplen)]

        if self.idx2dataset[i // self.cpi] not in ['train', 'val']:
            raise Exception('Invalid value when reading dataset partition')

        if self.split == 'TEST' and '-frcnn' in sys.argv:
            annotations = self.test_annotations
        else:
            annotations = self.train_annotations if self.idx2dataset[i // self.cpi] == 'train' else self.val_annotations
        
        img_fg, img_bg = separate_objects(img, caption_words, self.synonyms, annotations, self.idx2id[i // self.cpi], self.split == 'TEST')

        if i + 1 == len(self):
            print("Total of images in this dataset is {}".format(self.usableImages//5))

        if img_bg is None or img_fg is None:
            return None

        self.usableImages += 1

        if self.splitTransform is not None:
            img_bg = self.splitTransform(img_bg)
            img_fg = self.splitTransform(img_fg)

        if self.split is 'TRAIN':
            return img_id, img_fg, img_bg, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            return img_id, img_fg, img_bg, caption, caplen, all_captions

class CaptionDatasetFastText(CaptionDatasetSplit):
    def __init__(self, data_folder, data_name, split, transform=None, train_annotations=None, val_annotations=None):
        super().__init__(data_folder, data_name, split, transform, train_annotations, val_annotations)
        self.FTReplacer = FastTextReplacer(self.train_annotations, self.val_annotations, useFRCNN=self.split is 'TEST' and '-frcnn' in sys.argv)

    def __getitem__(self, i):
        data = super().__getitem__(i)
        if data is None:
            return None
        if self.split is 'TRAIN':
            img_id, img_fg, img_bg, caption, caplen = data
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            img_id, img_fg, img_bg, caption, caplen, all_captions = data
        if img_fg is None or img_bg is None:
            return None
        tensor_fg = torch.zeros([14, 14, 300])
        if '--bg' not in sys.argv:
            tensor_fg = self.FTReplacer.replace(img_id, tensor_fg, (None, None))
        if self.split is 'TRAIN':
            return tensor_fg, img_bg, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            return tensor_fg, img_bg, caption, caplen, all_captions


class CaptionDatasetFastTextWithReplacement(CaptionDatasetFastText):
    def __init__(self, data_folder, data_name, split, transform=None, train_annotations=None, val_annotations=None):
        super().__init__(data_folder, data_name, split, transform, train_annotations, val_annotations)
        if '--person' in sys.argv:
            print("Replacing only images with persons")
        self.simplifier = SentenceSimplifier()
        self.covMatrix = pickle.load(open('covMatrixAnns.pkl', 'rb'))
        self.id2classMap = getId2ClassMap()
        self.class2idMap = getClass2IdMap()
        self.rev_word_map = {v: k for k, v in self.word_map.items()}

    def __getitem__(self, i):
        data = super().__getitem__(i)
        if data is None:
            return None        
        if self.split is 'TRAIN':
            tensor_fg, img_bg, caption, caplen = data
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            tensor_fg, img_bg, caption, caplen, all_captions = data
        if tensor_fg is None or img_bg is None:
            return None

        img_id = self.idx2id[i // self.cpi]

        annotations = self.train_annotations if self.idx2dataset[i // self.cpi] == 'train' else self.val_annotations
        classesInImage = []
        annIds = annotations.getAnnIds(img_id)
        if len(annIds) == 0:
            raise Exception('Image ID {} without annotations'.format(img_id))
        anns = annotations.loadAnns(annIds)

        for annotation in anns:
            catinfo = annotations.loadCats(annotation['category_id'])[0]
            classesInImage.append(catinfo['name'])
    
        classesInImage = set(classesInImage)
        if '--person' in sys.argv:
            classCombinations = getClassCombinationsWithPerson(classesInImage)
            shuffle = False
        else:
            classCombinations = list(itertools.combinations(classesInImage, 2))
            shuffle = True
        random.shuffle(classCombinations)
        captionString = ' '.join([self.word_map[caption[idx].item()] for idx in range(caplen) if self.word_map[caption[idx].item()] != '<unk>'][1:-1])
        success, pair, synonymPair, simplifiedString = self.simplifier.simplify(captionString, classCombinations, shuffle)

        method = 'uniform'

        if success is True and random.random() < 0.5 and pair[0] in self.class2idMap.keys() and pair[1] in self.class2idMap.keys():
            wordToReplace = pair[1]
            replacementWord = wordToReplace
            while replacementWord == pair[0] or replacementWord == wordToReplace:
                if method == 'uniform':
                    replacementWord = self.id2classMap[np.random.choice(80, 1)[0]]
                elif method == 'multinomial':
                    clsIdx = self.class2idMap[pair[0]]
                    classProbabilities = 1 / (self.covMatrix[clsIdx, :] / np.sum(self.covMatrix[clsIdx, :]))
                    classProbabilities[clsIdx] = 0
                    classProbabilities = classProbabilities / np.sum(classProbabilities)
                    replacementWord = self.id2classMap[np.argmax(np.random.multinomial(1, classProbabilities))]
                else:
                    raise Exception('Invalid random choice method')
            clsPair = (wordToReplace, replacementWord)
            simplifiedString = re.sub(r"\b{}\b".format(synonymPair[1]), " {} ".format(replacementWord), simplifiedString)
            simplifiedString = re.sub(r" nt", "nt", simplifiedString)
            simplifiedString = simplifiedString.strip()

            tensor_fg = torch.zeros([14, 14, 300])
            img_id = self.idx2id[i // self.cpi]
            tensor_fg = self.FTReplacer.replace(img_id, tensor_fg, clsPair)

            caplen = torch.zeros(1)
            caption = torch.zeros(52)
            i = 0
            caption[i] = self.rev_word_map['<start>']
            i += 1
            for word in simplifiedString.split():
                if word in self.rev_word_map.keys():
                    caption[i] = self.rev_word_map[word]
                    i += 1
                else:
                    print('ERROR - word "{}" not found in word map'.format(word))
                    print('ERROR - Sentence with error is {}'.format(simplifiedString))
            caption[i] = self.rev_word_map['<end>']
            caplen[0] = i+1
            caption = caption.long()
            caplen = caplen.long()

        if self.split is 'TRAIN':
            return tensor_fg, img_bg, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            return tensor_fg, img_bg, caption, caplen, all_captions

class CaptionDatasetFastTextWithReplacementCV(CaptionDatasetFastText):
    def __init__(self, data_folder, data_name, split, transform=None, train_annotations=None, val_annotations=None):
        super().__init__(data_folder, data_name, split, transform, train_annotations, val_annotations)
        if '--person' in sys.argv:
            print("Replacing only images with persons")
        self.simplifier = SentenceSimplifier()
        self.covMatrix = pickle.load(open('covMatrixAnns.pkl', 'rb'))
        self.id2classMap = getId2ClassMap()
        self.class2idMap = getClass2IdMap()
        self.rev_word_map = {v: k for k, v in self.word_map.items()}

    def __getitem__(self, i):
        data = super().__getitem__(i)
        if data is None:
            return None        
        if self.split is 'TRAIN':
            tensor_fg, img_bg, caption, caplen = data
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            tensor_fg, img_bg, caption, caplen, all_captions = data
        if tensor_fg is None or img_bg is None:
            return None

        img_id = self.idx2id[i // self.cpi]
        
        captionString = ' '.join([self.word_map[caption[idx].item()] for idx in range(caplen) if self.word_map[caption[idx].item()] != '<unk>'][1:-1])
        annotations = self.train_annotations if self.idx2dataset[i // self.cpi] == 'train' else self.val_annotations

        classesInImage = []
        annIds = annotations.getAnnIds(img_id)
        if len(annIds) == 0:
            raise Exception('Image ID {} without annotations'.format(img_id))
        anns = annotations.loadAnns(annIds)

        for annotation in anns:
            catinfo = annotations.loadCats(annotation['category_id'])[0]
            classesInImage.append(catinfo['name'])
    
        classesInImage = set(classesInImage)
        if '--person' in sys.argv:
            classCombinations = getClassCombinationsWithPerson(classesInImage)
            shuffle = False
        else:
            classCombinations = list(itertools.combinations(classesInImage, 2))
            shuffle = True        
        random.shuffle(classCombinations)
        success, pair, synonymPair, simplifiedString = self.simplifier.simplify(captionString, classCombinations, shuffle)

        if success is True and pair[0] in self.class2idMap.keys() and pair[1] in self.class2idMap.keys():
            wordToReplace = pair[1]
            clsIdx = self.class2idMap[pair[0]]
            classProbabilities = 1 / (self.covMatrix[clsIdx, :] / np.sum(self.covMatrix[clsIdx, :]))
            classProbabilities[clsIdx] = 0
            classProbabilities = classProbabilities / np.sum(classProbabilities)
            replacementWord = wordToReplace
            while replacementWord == pair[0] or replacementWord == wordToReplace:
                replacementWord = self.id2classMap[np.argmax(np.random.multinomial(1, classProbabilities))]
            clsPair = (wordToReplace, replacementWord)

            self.covMatrix = updateCovMatrix(self.covMatrix, annotations, img_id, (self.class2idMap[clsPair[0]], self.class2idMap[clsPair[1]]))
            
            simplifiedString = re.sub(r"\b{}\b".format(synonymPair[1]), " {} ".format(replacementWord), simplifiedString)
            simplifiedString = re.sub(r" nt", "nt", simplifiedString)
            simplifiedString = simplifiedString.strip()

            tensor_fg = torch.zeros([14, 14, 300])
            tensor_fg = self.FTReplacer.replace(img_id, tensor_fg, clsPair)

            caplen = torch.zeros(1)
            caption = torch.zeros(52)
            i = 0
            caption[i] = self.rev_word_map['<start>']
            i += 1
            for word in simplifiedString.split():
                if word in self.rev_word_map.keys():
                    caption[i] = self.rev_word_map[word]
                    i += 1
                else:
                    print('ERROR - word "{}" not found in word map'.format(word))
                    print('ERROR - Sentence with error is {}'.format(simplifiedString))
            caption[i] = self.rev_word_map['<end>']
            caplen[0] = i+1
            caption = caption.long()
            caplen = caplen.long()
            

        if self.split is 'TRAIN':
            return tensor_fg, img_bg, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            return tensor_fg, img_bg, caption, caplen, all_captions  

class CaptionDatasetFastTextStrangeImages(CaptionDatasetFastText):
    def __init__(self, data_folder, data_name, split, transform=None, train_annotations=None, val_annotations=None):
        super().__init__(data_folder, data_name, split, transform, train_annotations, val_annotations)
        self.simplifier = SentenceSimplifier()
        self.covMatrix = pickle.load(open('covMatrixAnns.pkl', 'rb'))
        self.id2classMap = getId2ClassMap()
        self.class2idMap = getClass2IdMap()
        self.rev_word_map = {v: k for k, v in self.word_map.items()}

        if '--mkjson' in sys.argv:
            print("Generating JSON file for Unusual image dataset")
            #Custom coco json for CHAIR evaluation
            self.instGt = json.load(open('dataset/annotations/instances_train2014.json', 'r'))
            self.captionGt = json.load(open('dataset/annotations/captions_train2014.json', 'r'))

            self.trainDictInstances = {}
            self.trainDictCaptions = {}
            self.valDict = {}
            self.trainDictInstances = {}
            self.trainDictInstances['info'] = self.instGt['info']
            self.trainDictInstances['licenses'] = self.instGt['licenses']
            self.trainDictInstances['images'] = []
            self.trainDictInstances['annotations'] = []
            self.trainDictInstances['categories'] = self.instGt['categories']
            self.trainDictCaptions['info'] = self.captionGt['info']
            self.trainDictCaptions['licenses'] = self.captionGt['licenses']
            self.trainDictCaptions['images'] = []
            self.trainDictCaptions['annotations'] = []
            self.valDict['info'] = self.instGt['info']
            self.valDict['licenses'] = self.instGt['licenses']
            self.valDict['images'] = []
            self.valDict['annotations'] = []
            self.valDict['categories'] = self.instGt['categories']
            self.currAnnId = 0

    def __getitem__(self, i):
        data = super().__getitem__(i)
        if data is None:
            return None        
        if self.split is 'TRAIN':
            tensor_fg, img_bg, caption, caplen = data
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            tensor_fg, img_bg, caption, caplen, all_captions = data
        if tensor_fg is None or img_bg is None:
            return None

        img_id = self.idx2id[i // self.cpi]
        
        random.seed(i)
        
        captionString = ' '.join([self.word_map[caption[idx].item()] for idx in range(caplen) if self.word_map[caption[idx].item()] != '<unk>'][1:-1])
        if self.split == 'TEST' and '-frcnn' in sys.argv:
            annotations = self.test_annotations
        else:
            annotations = self.train_annotations if self.idx2dataset[i // self.cpi] == 'train' else self.val_annotations
        
        classesInImage = []
        annIds = annotations.getAnnIds(img_id)
        if len(annIds) == 0:
            raise Exception('Image ID {} without annotations'.format(img_id))
        anns = annotations.loadAnns(annIds)

        for annotation in anns:
            catinfo = annotations.loadCats(annotation['category_id'])[0]
            classesInImage.append(catinfo['name'])
    
        classesInImage = list(set(classesInImage))
        classesInImage.sort()
        classCombinations = list(itertools.combinations(classesInImage, 2))
        random.shuffle(classCombinations)
        success, pair, synonymPair, simplifiedString = self.simplifier.simplifyStrange(captionString, classCombinations)

        if success is True and pair[0] in self.class2idMap.keys() and pair[1] in self.class2idMap.keys():
            wordToReplace = pair[1]
            clsIdx = self.class2idMap[pair[0]]
            sortedClasses = np.argsort(self.covMatrix[clsIdx, :])
            uncommonClasses = sortedClasses[0:10]
            replacementWord = wordToReplace
            while replacementWord == wordToReplace or replacementWord == pair[0]:
                replacementWord = self.id2classMap[random.choice(uncommonClasses)]
            clsPair = (wordToReplace, replacementWord)
            
            simplifiedString = re.sub(r"\b{}\b".format(synonymPair[1]), " {} ".format(replacementWord), simplifiedString)
            simplifiedString = re.sub(r" nt", "nt", simplifiedString)
            simplifiedString = re.sub(r"  ", " ", simplifiedString)
            simplifiedString = simplifiedString.strip()

            if '--mkjson' in sys.argv:
                for annotation in anns:
                    catName = annotations.loadCats(annotation['category_id'])[0]['name']
                    annCpy = dict(annotation)
                    annCpy['category_id'] = annCpy['category_id'] if catName != wordToReplace else list(annotations.cats.keys())[self.class2idMap[replacementWord]]
                    annCpy['id'] = self.currAnnId
                    self.trainDictInstances['annotations'].append(annCpy)
                    self.currAnnId += 1
                self.trainDictCaptions['annotations'].append({
                    "image_id": img_id,
                    "id": self.currAnnId,
                    "caption": simplifiedString
                })
                self.currAnnId += 1
                imgInfo = {
                    "license": 4,
                    "file_name": "",
                    "coco_url": "",
                    "height": 0,
                    "width": 0,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": img_id
                }
                self.trainDictCaptions['images'].append(imgInfo)
                self.trainDictInstances['images'].append(imgInfo)
                if i > 24760:
                    with (open('dataset/annotations/unusual/instances_train2014.json', 'w')) as f:
                        json.dump(self.trainDictInstances, f)
                    with (open('dataset/annotations/unusual/captions_train2014.json', 'w')) as f:
                        json.dump(self.trainDictCaptions, f)
                    with (open('dataset/annotations/unusual/instances_val2014.json', 'w')) as f:
                        json.dump(self.valDict, f)
                    with (open('dataset/annotations/unusual/captions_val2014.json', 'w')) as f:
                        json.dump(self.valDict, f)

            if i % 5 == 0:
                print('Image ID: {}'.format(img_id))
                print('Strange Image - {}'.format(simplifiedString))

            tensor_fg = torch.zeros([14, 14, 300])
            tensor_fg = self.FTReplacer.replace(img_id, tensor_fg, clsPair)

            caplen = torch.zeros(1)
            caption = torch.zeros(52)
            i = 0
            caption[i] = self.rev_word_map['<start>']
            i += 1
            for word in simplifiedString.split():
                if word in self.rev_word_map.keys():
                    caption[i] = self.rev_word_map[word]
                    i += 1
                else:
                    print('ERROR - word "{}" not found in word map'.format(word))
                    print('ERROR - Sentence with error is {}'.format(simplifiedString))
            caption[i] = self.rev_word_map['<end>']
            caplen[0] = i+1
            caption = caption.long()
            caplen = caplen.long()
            
            if self.split is 'TRAIN':
                return tensor_fg, img_bg, caption, caplen
            else:
                # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
                return tensor_fg, img_bg, caption, caplen, all_captions
        return None

class FasterRCNNDataset(CaptionDataset):
    def __init__(self, data_folder, data_name, split, transform=None, train_annotations=None, val_annotations=None):
        super().__init__(data_folder, data_name, split, None, True)
        if train_annotations is None:
            self.train_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_train2014.json'))
        else:
            self.train_annotations = train_annotations
        if val_annotations is None:
            self.val_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_val2014.json'))
        else:
            self.val_annotations = val_annotations
        self.idx2dataset = {}
        
        with(open(os.path.join(data_folder, self.split + '_ids.txt'), 'r')) as f:
            for i, line in enumerate(f):
                values = line.rstrip().split()
                self.idx2dataset[i] = values[1]

    def __getitem__(self, i):
        img_id, _, img = super().__getitem__(i)
        annotations = self.train_annotations if self.idx2dataset[i // self.cpi] == 'train' else self.val_annotations
        imgInfo = annotations.loadImgs(img_id)
        img_h = imgInfo[0]['height']
        img_w = imgInfo[0]['width']
        return img_id, img, img_h, img_w

        
