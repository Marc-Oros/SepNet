import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from image_manipulation import separate_objects
from utils import get_word_synonyms
from pycocotools.coco import COCO


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
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

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        # Load word map (word2ix)
        with open('dataset/output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json', 'r') as j:
            word_map = json.load(j)
        self.word_map = {v: k for k, v in word_map.items()}  # ix2word
        self.synonyms = get_word_synonyms()

        self.train_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_train2014.json'))
        self.val_annotations = COCO(os.path.join('dataset', 'annotations', 'instances_val2014.json'))
        self.idx2id = {}
        with(open(os.path.join(data_folder, self.split + '_ids.txt'), 'r')) as f:
            for i, line in enumerate(f):
                self.idx2id[i] = int(line)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        caption_words = [self.word_map[caption[idx].item()] for idx in range(caplen)]
        
        img_fg, img_bg = separate_objects(img, caption_words, self.synonyms, self.train_annotations, self.val_annotations, self.idx2id[i // self.cpi])

        if self.transform is not None:
            img_bg = self.transform(img_bg)
            img_fg = self.transform(img_fg)

        if self.split is 'TRAIN':
            return img_fg, img_bg, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img_fg, img_bg, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
