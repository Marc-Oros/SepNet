import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
from pycocoevalcap.eval import COCOEvalCap
from PIL import Image

# Parameters
data_folder = 'dataset/output/'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = 'dataset/output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

train_annotations=COCO(os.path.join('dataset', 'annotations', 'instances_train2014.json'))
val_annotations=COCO(os.path.join('dataset', 'annotations', 'instances_val2014.json'))

FTReplacer = FastTextReplacer(train_annotations, val_annotations)

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # Test Dataset
    testDataset = CaptionDatasetFastText(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize]), train_annotations=train_annotations, val_annotations=val_annotations)
    # DataLoader
    loader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=my_collate_test)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    #Text file with captions
    captionOutFile = open('evalCaptionsReplaced.txt', 'w')
    imgList = [27, 367, 1024, 3089, 5048, 6532, 7192, 12121, 18999, 24240]
    replacements = {
        27: ("dining table", "elephant"),
        367: ("person", "horse"),
        1024: ("frisbee", "pizza"),
        3089: ("pizza", "hot dog"),
        5048: ("bicycle", "horse"),
        6532: ("person", "dog"),
        7192: ("carrot", "broccoli"),
        12121: ("person", "bear"),
        18999: ("microwave", "tv"),
        24240: ("bird", "cat")
    }

    # For each image
    for i, (tensor_fg, img_bg, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        #Only generate one caption per image, a limitation of the coco evaluation code (only one result per id)
        if i not in imgList:
            continue
        
        imgId = testDataset.getImgId(i)
        
        if tensor_fg is None:
            continue

        img_bg_arr = img_bg[0].permute(1, 2, 0).numpy()
        img_bg_img = Image.fromarray((img_bg_arr * 255).astype(np.uint8))
        img_bg_img.save('imgs_replace/img_bg_{}.jpg'.format(imgId))

        k = beam_size

        replacementTensor = torch.zeros([14, 14, 300])

        replacementTensor = FTReplacer.replace(imgId, replacementTensor, replacements[i])

        tensor_fg[0] = replacementTensor

        # Move to GPU device, if available
        tensor_fg = tensor_fg.to(device)  # (1, 3, 256, 256)
        img_bg = img_bg.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(tensor_fg, img_bg)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        if len(complete_seqs_scores) == 0:
            print("Skipping item with no scores")
            continue

        seqIdx = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[seqIdx]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypo = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        hypotheses.append(hypo)
        captionWords = [rev_word_map[item] for item in hypo]
        captionString = ' '.join(captionWords)

        assert len(references) == len(hypotheses)

        #Print some captions and their image ids
        captionOutFile.write('Image idx: {}\n'.format(i))
        captionOutFile.write('Image ID: {}\n'.format(imgId))
        captionOutFile.write('Replacement item: {} -> {}\n'.format(replacements[i][0], replacements[i][1]))
        captionOutFile.write('Caption: {}\n'.format(captionString))

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
