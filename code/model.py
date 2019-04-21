import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as models

from pycocotools.coco import COCO

'''
feature extractor Resnet-152 pretrained on ImageNet
uses all layers except for last fc layer
output feature dimensions = ftsize
'''
class ImageEnc(nn.Module):
    def __init__(self, ftsize):
        super(ImageEnc, self).__init__()
        resnet = models.resnet152(pretrained=True)
        layers = list(resnet.children())[:-1]
        self.model = nn.Sequential(*layers)
        self.linear = nn.Linear(resnet.fc.in_features, ftsize)
        self.bn = nn.BatchNorm1d(ftsize, momentum=0.01)
        
    def forward(self, x):
        # (bs x 3 x H x W) -> (bs x 2048 x 1 x 1)
        with torch.no_grad():
            resout = self.model(x)
        # -> (bs x 2048)
        resout = resout.reshape(resout.size(0), -1)
        # -> (bs x ftsize)
        out = self.bn(self.linear(resout))
        return out

'''
caption generator from features
uses LSTM
embedding size must equal feature size!!
'''
class CaptionGen(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(CaptionGen, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # input, (h0, c0) -> output, (hn, cn)
        # input : (bs x seqlen x embed_size) or PackedSequence
        # output : (bs x seqlen x hidden_size) or PackedSequence
        # h0, c0, hn, cn : (num_layers x bs x hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        
    # features : (bs x ftsize), captions : (bs x padded_len), lengths : (bs)
    # captions MUST be padded, must be index of words in dictionary
    def forward(self, features, captions, lengths):
        # (bs x padded_len) -> (bs x padded_len x embed_size)
        embedded = self.embed(captions)
        # (bs x 1 x embed_size) & (bs x padded_len x embed_size) -> (bs x (1+padded_len) x embed_size)
        embedded = torch.cat((features.unsqueeze(1), embedded), 1)
        # -> PackedSequence
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        # hiddens : PackedSequence
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        # output is packed in order to compare directly with annotations for computing loss
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for _ in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
