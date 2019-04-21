import nltk
import pickle
from collections import Counter
from pycocotools.coco import COCO

class Vocabulary():
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(jsonpath, threshold):
    coco = COCO(jsonpath)
    counter = Counter()
    annids = coco.anns.keys()
    for i, annid in enumerate(annids):
        caption = str(coco.anns[annid]['caption'])
        for word in caption.split(' '):
            word = word.strip('.,\'"!?').lower()
            if len(word) > 0:
                counter.update([word])
        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(annids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocabulary = Vocabulary()
    vocabulary.add_word('<pad>')
    vocabulary.add_word('<BOS>')
    vocabulary.add_word('<EOS>')
    vocabulary.add_word('<unk>')

    for i, word in enumerate(words):
        vocabulary.add_word(word)
    return vocabulary

def main():
    annfile = '../../../datasets/mscoco/annotations/captions_train2017.json'
    pklfile = '../data/vocab_train2017.pkl'
    vocab = build_vocab(jsonpath=annfile, threshold=3)
    with open(pklfile, 'wb') as f:
        pickle.dump(vocab, f)
    print('total vocab size: {}'.format(len(vocab)))
    print('Saved vocab wrapper to {}'.format(pklfile))

if __name__ == '__main__':
    main()
