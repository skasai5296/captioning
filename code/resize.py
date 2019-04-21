import os
from PIL import Image

def main():
    inpath = '../../../datasets/mscoco/train2017'
    outpath = '../../coco/dataset'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for f in os.listdir(inpath):
        im = Image.open(os.path.join(inpath, f))
        im_resized = im.resize((256, 256))
        im_resized.save(os.path.join(outpath, f))

if __name__ == '__main__':
    main()
