import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import pickle

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

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def generate():
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=my_collate)

    k = 1
    captions = np.zeros((len(loader), 52), dtype=np.int64)
    capLengths = np.zeros(len(loader))

    first_caption_idx = 0

    # For each image
    for i, (img_fg, img_bg, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="Generating captions...")):

        if i != 0 and i % 5 == 0:
            first_caption_idx += 5

        if img_bg is None or img_fg is None:
            continue

        # Move to GPU device, if available
        img_fg = img_fg.to(device)  # (1, 3, 256, 256)
        img_bg = img_bg.to(device)  # (1, 3, 256, 256)


        # Encode
        encoder_out = encoder(img_fg, img_bg)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        captions[i, 0] = word_map['<start>']
        lastWord = torch.LongTensor([word_map['<start>']]).to(device)

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(lastWord).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            _ , word_idx_list = scores[0].topk(5, 0, True, True)

            curr_pick = 0

            word_idx = word_idx_list[curr_pick]

            lastWord = torch.LongTensor([word_idx]).to(device)
            captions[i, step] = word_idx.item()

            #Logic to generate multiple captions
            for offset in range(5):
                if offset == i % 5:
                    continue
                same_caption = True
                for word_num in range(step):
                    if captions[first_caption_idx+offset, word_num+1] != captions[i, word_num+1]:
                        same_caption = False
                        break
                if same_caption is True:
                    curr_pick += 1
                    word_idx = word_idx_list[curr_pick]
                    lastWord = torch.LongTensor([word_idx]).to(device)
                    captions[i, step] = word_idx.item()

            if word_idx == word_map['<end>']:
                break

            # Break if things have been going on too long
            if step > 50:
                print("incomplete sequence")
                break
            step += 1
        capLengths[i] = step

    #pickle.dump(capLengths, open("caplensSplit.pkl", "wb"))
    pickle.dump(captions, open("captionsSplitFromAnns.pkl", "wb"))
    return


if __name__ == '__main__':
    generate()
