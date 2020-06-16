from datasets import CaptionDataset
from tqdm import tqdm
import os
import pickle
import json

"""train = CaptionDataset('dataset/output',
                         'coco_5_cap_per_img_5_min_word_freq',
                         'TRAIN',
                         None)

val = CaptionDataset('dataset/output',
                         'coco_5_cap_per_img_5_min_word_freq',
                         'VAL',
                         None)"""

test = CaptionDataset('dataset/output',
                         'coco_5_cap_per_img_5_min_word_freq',
                         'TEST',
                         None)

idx2id = {}
with(open(os.path.join('dataset', 'output','TEST_ids.txt'), 'r')) as f:
    for i, line in enumerate(f):
        values = line.rstrip().split()
        idx2id[i] = int(values[0])

"""id2idx = {value : key for (key, value) in idx2id.items()}"""

captions = pickle.load(open('captionsOriginal.pkl', 'rb'))

word_map_file = 'dataset/output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

for dataset in [test]:
    for i, values in enumerate(dataset):
        if i > 250:
            break
        if i % 5 != 0:
            continue

        if values is None:
            continue

        l = 0
        sentence = ''
        while captions[i, l] != 0:
            sentence += ' {}'.format(rev_word_map[captions[i, l]])
            l += 1
        print('Image with id {}'.format(idx2id[i//5]))
        print(sentence)

exit()
