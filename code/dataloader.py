import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

from pycocotools.coco import COCO

class CocoDataset(Dataset):
    def __init__(self, imgpath, annpath, vocab, transform=None):
        self.imgpath = imgpath
        self.coco = COCO(annpath)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, idx):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[idx]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.imgpath, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = []
        for word in str(caption.split(' ')):
            word = word.strip('.,\'"!?').lower()
            if len(word) > 0:
                tokens.append(word)
        
        caption = []
        caption.append(vocab('<BOS>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<EOS>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(imgpath, annpath, vocab, transform, bs, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(imgpath=imgpath,
                       annpath=annpath,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (bs, 3, 224, 224).
    # captions: a tensor of shape (bs, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=bs,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
