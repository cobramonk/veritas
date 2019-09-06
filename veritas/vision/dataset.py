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


#__all__ = ['ImageDataset', 'ImageDataPack' ]


#########################################################
## functools ##
#########################################################

def readerFileClassif(folder, suffix, x, y):
    path = Path(folder)/ ( x + suffix )
    return Image.open(path).convert('RGB'), y

def readerFileDetection(*args):
    return readerFileClassif(*args)

def readerFileSuperRes(xfolder, yfolder, suffix, x, y):
    return (Image.open( Path(xfolder)/ (x+suffix)).convert('RGB'),
            Image.open( Path(yfolder)/ (y+suffix)).convert('RGB'))

def readerMongoClassif(dbroute, db, collection, imgKey, x, y):
    x = list(dbQuery(dbroute, db, collection, {'_id': x}, {imgKey:1}) )[0][imgKey]
    return Image.open(io.BytesIO(x)).convert('RGB'), y

def readerMongoDetection(*args):
    return readerMongoClassif(*args)


def preprocImage(imSize, x, transformVals):
    x = TF.resize(x, (imSize[1], imSize[0]))
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
    imOrgSize = x.size # get size of image before preproc
    x = preprocImage(imSize, x, transformVals)
    y = preprocDetectionTarg(imSize,y, transformVals,imOrgSize)
    return x,y

def preprocSuperRes(imSize, x, y, transformVals):
    return ( preprocImage(imSize, x, transformVals),
            preprocImage(imSize, y, transformVals) )


datasetType = {'classification':preprocClassif, 'detection': preprocDetection, 'superres': preprocSuperRes};

###########################################################
## dataset classes ##
###########################################################


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

class ClassifDataPack(ImageDataPack):
    preproc = preprocClassif
    readerFile = readerFileClassif
    readerMongo = readerMongoClassif

    @classmethod
    def fromSingleList(cls, x,y, split_ratio = [0.8, 0.2], folder = '.', imSize = (224,224), suffix='.jpg', bs=2, transforms = [] ):
        trn_ds, val_ds, test_ds = None, None, None
        Xs, Ys = splitByRatio(x,y, split_ratio)
        return cls.fromSplitLists(Xs, Ys, [folder]*3, suffix, imSize, bs,
                transforms)

    @classmethod
    def fromSingleMongo(cls, x, y, dbroute, db, collection, imgKey='img', split_ratio = [0.8,0.2], imSize=(224,224), bs=2, transforms = []):
        Xs, Ys = splitByRatio(x,y, split_ratio)
        return cls.fromSplitMongo(Xs, Ys, dbroute, db, [collection]*3, imgKey, imSize,
            bs, transforms)

    @classmethod
    def fromSplitLists(cls, Xs, Ys, folders, suffix='.jpg', imSize = (224,224), bs =2,
            transforms = []):
        dRs = [ partial(cls.readerFile, folder, suffix) for folder in folders]
        preproc = partial(cls.preproc, imSize)
        return cls.fromReaderAndPreproc(Xs, Ys, dRs, [preproc]*3, [transforms, [],[]], bs )

    @classmethod
    def fromSplitMongo(cls, Xs, Ys, dbroute = None, db = None, collections = [], imgKey = 'img',
            imSize = (224, 224), bs =2, transforms =[]):
        dRs = [partial (cls.readerMongo, dbroute, db, collection, imgKey) for collection in collections]
        preproc = partial(cls.preproc, imSize)
        return cls.fromReaderAndPreproc(Xs, Ys, dRs, [preproc]*3, [transforms, [],[]], bs )



class DetectionDataPack(ClassifDataPack):
    preproc = preprocClassif
    readerFile = readerFileClassif
    readerMongo = readerMongoClassif

class SuperResDataPack(ImageDataPack):
    preproc = preprocSuperRes
    readerFile = readerFileSuperRes

    @classmethod
    def fromSingleList(cls, x, y, split_ratio = [0.8, 0.2], xfolder = 'small',
            yfolder = 'large', imSize=(256,256), suffix='.jpg', bs=2, transforms=[]):
        Xs, Ys = splitByRatio(x,y, split_ratio)
        return cls.fromSplitLists(Xs, Ys, [xfolder]*3, [yfolder]*3, suffix,
                imSize, bs, transforms)

    @classmethod
    def fromSplitLists(cls, Xs, Ys, xfolders, yfolders, suffix = '.jpg', imSize=(256, 256), bs=2, transforms=[]):
        dRs = [ partial(cls.readerFile, xfolder, yfolder, suffix) for xfolder,yfolder
                in zip(xfolders, yfolders)]
        preproc = partial(cls.preproc, imSize)
        return cls.fromReaderAndPreproc(Xs, Ys, dRs, [preproc]*3, [transforms, [], []], bs)

