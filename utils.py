import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from torch.utils.data.dataloader import default_collate


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with open(os.path.join(output_folder, split + '_ids' + '.txt'), 'a') as f:
            for impath in impaths:
                #Extraction of the image id from the file name
                image_id = int(impath.split('/')[-1].split('_')[-1].split('.')[0])
                datasetOriginalPartition = impath.split('/')[1][:-4]
                f.write('{} {}\n'.format(image_id, datasetOriginalPartition))        

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def get_word_synonyms():
    synonyms = []
    with(open('synonyms.txt', 'r')) as f:
        for line in f:
            synonym_items = line.rstrip().split(', ')
            synonyms.append(synonym_items)
    return synonyms

def getClsList():
    clsList = [
        'person',
        'bicycle',
        'car',
        'motorcycle',
        'airplane',
        'bus',
        'train',
        'truck',
        'boat',
        'traffic light',
        'fire hydrant',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'cow',
        'elephant',
        'bear',
        'zebra',
        'giraffe',
        'backpack',
        'umbrella',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'dining table',
        'toilet',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush'
        ]
    return clsList

def getId2ClassMap():
    clsList = getClsList()
    clsMap = {i: clsName for i, clsName in enumerate(clsList)}
    clsMap[-1] = None
    return clsMap

def getClass2IdMap():
    clsList = getClsList()
    clsMap = {clsName: i for i, clsName in enumerate(clsList)}
    return clsMap


def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    if len(batch) < 1:
        return None, None, None, None

    return default_collate(batch)

def my_collate_test(batch):
    batch = list(filter(lambda x : x is not None, batch))
    if len(batch) < 1:
        return None, None, None, None, None

    return default_collate(batch)

def getProbabilityValues():
    #Probabilities of classes for person
    return np.array(
        [0,
        0.002489667842,
        0.00077443315,
        0.002257435451,
        0.007726152318,
        0.002185485317,
        0.003698135829,
        0.001652279562,
        0.003164750853,
        0.002641525592,
        0.009567851417,
        0.01022155555,
        0.02350957777,
        0.001562839928,
        0.00875356619,
        0.01048197735,
        0.003245898311,
        0.003265219134,
        0.01662293378,
        0.01119503703,
        0.008396277774,
        0.08661423388,
        0.06856960182,
        0.01394635969,
        0.001296824621,
        0.001898120466,
        0.001064469886,
        0.001721412598,
        0.003809422324,
        0.003457290848,
        0.002197156801,
        0.004166254288,
        0.001540889929,
        0.003116800083,
        0.002689003993,
        0.002543540099,
        0.002067425181,
        0.001734110057,
        0.001802486795,
        0.001394635969,
        0.004063383812,
        0.001391099276,
        0.004013830351,
        0.002847180699,
        0.004063383812,
        0.002276169355,
        0.007062963278,
        0.01456345525,
        0.006609118248,
        0.013489102,
        0.02789271939,
        0.01732284678,
        0.008269700722,
        0.003748679826,
        0.007654281134,
        0.004252378408,
        0.0008126767624,
        0.003170848639,
        0.003400145545,
        0.004927157017,
        0.001139660972,
        0.01431017777,
        0.003317883959,
        0.003508892204,
        0.0125623698,
        0.003304559124,
        0.01111939489,
        0.001650622311,
        0.01456345525,
        0.006428400171,
        0.3291340888,
        0.006885650392,
        0.007950098762,
        0.00298669772,
        0.003164750853,
        0.007583734764,
        0.01732284678,
        0.008106750955,
        0.05142720137,
        0.01265900341], dtype=np.float)

def updateCovMatrix(covMatrix, annotations, img_id, pair):
    classesInImage = []
    annIds = annotations.getAnnIds(img_id)
    if len(annIds) == 0:
        raise Exception('Image ID {} without annotations'.format(img_id))
    anns = annotations.loadAnns(annIds)

    for annotation in anns:
        catinfo = annotations.loadCats(annotation['category_id'])[0]
        classesInImage.append(catinfo['name'])
    
    classesInImage = set(classesInImage)

    clsMap = getClass2IdMap()
    
    for cls_name in classesInImage:
        clsId = clsMap[cls_name]
        if clsId not in pair:
            covMatrix[clsId, pair[0]] = max(covMatrix[clsId, pair[0]] - 1, 1)
            covMatrix[pair[0], clsId] = max(covMatrix[pair[0], clsId] - 1, 1)
            covMatrix[clsId, pair[1]] = max(covMatrix[clsId, pair[1]] + 1, 1)
            covMatrix[pair[1], clsId] = max(covMatrix[pair[1], clsId] + 1, 1)
    return covMatrix

def getClassTotalsFromCV(covMatrix):
    nClasses = covMatrix.shape[0]
    result = np.zeros(nClasses)
    for i in range(nClasses):
        result[i] = np.sum(covMatrix[i,:])
    return result

def getClassProbabilites(covMatrix, classTotals):
    probs = np.zeros((covMatrix.shape[0], covMatrix.shape[1]))
    for i in range(covMatrix.shape[0]):
        if np.sum(covMatrix[i]) != classTotals[i]:
            raise Exception("Class totals don't match the covariance matrix")
        probs[i, :] = 1 / (covMatrix[i, :] / classTotals[i])
        probs[i, :] = probs[i, :] / np.sum(probs[i, :])
    return probs
