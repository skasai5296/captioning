import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from model import ImageEnc, CaptionGen
from vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_image(impath, transform=None):
    im = Image.open(impath)
    im = im.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        im_ten = transform(im).unsqueeze(0)
    return im_ten

def main(args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    encoder = ImageEnc(args.embed_size).eval()
    decoder = CaptionGen(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    im_ten = get_image(args.image_path, transform=transform)
    im_ten = im_ten.to(device)
    features = encoder(im_ten)
    ids = decoder.sample(features)
    ids = ids.cpu().numpy().reshape((-1))
    words = []
    for i in ids:
        words.append(vocab.idx2word[i])
    sentence = ' '.join(words)
    print(sentence)
    image = Image.open(args.image_path)
    plt.imshow(np.asarray(image))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='test.jpg', help='path for image')
    parser.add_argument('--encoder_path', type=str, default='../data/model/encoder-1-4000.ckpt', help='path for encoder')
    parser.add_argument('--decoder_path', type=str, default='../data/model/decoder-1-4000.ckpt', help='path for decoder')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab_train2017.pkl', help='path for vocabulary')
    parser.add_argument('--embed_size', type=int, default=256, help='size of feature vector')
    parser.add_argument('--hidden_size', type=int, default=512, help='size of LSTM memory')
    parser.add_argument('--num_layers', type=int, default=1, help='layers in LSTM')
    args = parser.parse_args()
    main(args)
