import torch
import numpy as np
from ..basics import BasicDataset, BasicDataPack
from pathlib import Path
from .utils import *
from torch.utils import data as dd
from functools import partial
import json
import io

# bad code
PADLEN = 100


class ImageDataset(dd.Dataset):
    def __init__(self,x,y, dataReader, preproc, transforms = []):
        self.examples = list(zip(x,y))
        self.dataReader = dataReader
        self.preproc = preproc
        self.transforms = transforms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,index):
        transformVals = transformValGen(self.transforms)
        x, y = self.dataReader(*self.examples[index])
        return self.preproc(x, y, transformVals)

class ImageDataPack(BasicDataPack):

    @classmethod
    def fromReaderAndPreproc(cls, X, Y, dRs, preprocs, transforms, bs):
        return cls(*[ImageDataset(X[i], Y[i], dRs[i], preprocs[i], transforms[i])
                if X[i] else None for i in range(len(X)) ], bs)

    @classmethod
    def fromSingleList(cls, x,y, split_ratio = [0.8, 0.2], folder = '.', imSize = (224,224), suffix='.jpg', bs=2, transforms = [], dstype = 'classification'):
        trn_ds, val_ds, test_ds = None, None, None
        splitIdxs = [ int(split_ratio[0] * len(x)), int((split_ratio[0]+split_ratio[1]) * len(x)) ]
        Xs = [  x[:splitIdxs[0]], x[splitIdxs[0]:splitIdxs[1]], x[splitIdxs[1]:]]
        Ys = [  y[:splitIdxs[0]], y[splitIdxs[0]:splitIdxs[1]], y[splitIdxs[1]:]]
        return cls.fromSplitLists(Xs, Ys, [folder]*3, suffix, imSize, bs,
                transforms, dstype  )

    @classmethod
    def fromSingleMongo(cls, x, y, dbroute, db, collection, imgKey='img', split_ratio = [0.8,0.2], imSize=(224,224), bs=2, transforms = [], dstype='classification' ):
        splitIdxs = [ int(split_ratio[0] * len(x)), int((split_ratio[0]+split_ratio[1]) * len(x)) ]
        Xs = [  x[:splitIdxs[0]], x[splitIdxs[0]:splitIdxs[1]], x[splitIdxs[1]:]]
        Ys = [  y[:splitIdxs[0]], y[splitIdxs[0]:splitIdxs[1]], y[splitIdxs[1]:]]
        return cls.fromSplitMongo(Xs, Ys, dbroute, db, [collection]*3, imgKey, imSize,
            bs, transforms, dstype  )

    @classmethod
    def fromSplitLists(cls, Xs, Ys, folders, suffix='.jpg', imSize = (224,224), bs =2, transforms = [], dstype= 'classification' ):
        dRs = [ partial(readerFileClassif, folder, suffix) for folder in folders ]
        preproc = preprocClassif if dstype == 'classification' else preprocDetection
        preproc = partial(preproc, imSize)
        return cls.fromReaderAndPreproc(Xs, Ys, dRs, [preproc]*3, [transforms, [],[]], bs )

    @classmethod
    def fromSplitMongo(cls, Xs, Ys, dbroute = None, db = None, collections = [], imgKey = 'img',
            imSize = (224, 224), bs =2, transforms =[], dstype='classification'):
        dRs = [partial (readMongoClassif, dbroute, db, collection, imgKey) for collection in collections]
        preproc = preprocClassif if dstype == 'classification' else preprocDetection
        preproc = partial(preproc, imSize)
        return cls.fromReaderAndPreproc(Xs, Ys, dRs, [preproc]*3, [transforms, [],[]], bs )



class DetectionDataPack(ImageDataPack):
    pass

__all__ = ['ImageDataset', 'ImageDataPack' ]


#########################################################
## functools ##
#########################################################

def readerFileClassif(folder, suffix, x, y):
    path = Path(folder)/ ( x + suffix )
    return Image.open(path), y

def readerFileDetection(folder, suffix, x, y):
    return readerFileClassif(folder, suffix, x, y)

def readMongoClassif(dbroute, db, collection, imgKey, x, y):
    x = list(dbQuery(dbroute, db, collection, {'_id': x}, {imgKey:1}) )[0][imgKey]
    return Image.open(io.BytesIO(x)), y


def preprocImage(imSize, x, transformVals):
    x = TF.resize(x, imSize)
    x = imageTransform(x, transformVals)
    x = imageTensor(x)
    return x

def preprocDetectionTarg( imSize, y, transformVals, imOrgSize):
    y = np.float32(y)
    y = targResize(y, imSize, imOrgSize)
    y = targTransform(y, imSize, transformVals)
    y = targPad(y,PADLEN)
    y = torch.tensor(y).float()
    return y

def preprocClassif(imSize, x, y, transformVals):
    x = preprocImage(imSize, x, transformVals)
    return x,y

def preprocDetection(imSize, x, y, transformVals):
    imOrgSize = x.size
    x = preprocImage(imSize, x, transformVals)
    y = preprocDetectionTarg(imSize,y, transformVals,imOrgSize)
    return x,y

