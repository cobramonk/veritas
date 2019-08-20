import torch
import numpy as np
from ..basics import BasicDataset, BasicDataPack
from pathlib import Path
from ..utils.image import tensorImage, transformValGen, tensorY


class ClassDataset(BasicDataset):
    """
    __init__ params:
    'x' - list of filepaths
    'y' - list of targets, each entry can be a single number representing the class or an array containing left, top, right, bottom coordiantes and classes
    'folder' -  path to the folder contaiting the images.
    'imSize' - a tuple determining the dimensions of the image.
     """
    def __init__(self,x, y, folder, imSize = (224,224), suffix='.jpg' ,transforms = []):
        super().__init__(x,y)
        self.imSize = imSize
        self.folder = folder
        self.suffix = suffix
        self.transforms = transforms

    def __getitem__(self, index):
        transformVals = transformValGen(self.transforms)
        return self.preprocX(index, transformVals), self.preprocY(index, transformVals)

    def filePath(self, index):
        return Path(self.folder) / ( self.x[index] + self.suffix )

    def preprocX(self, index, transformVals):
        return tensorImage( self.filePath(index) , self.imSize, transformVals)

    def preprocY(self, index, transformVals):
        return torch.tensor(self.y[index])

class ClassDataPack(BasicDataPack):
    DS = ClassDataset

    @classmethod
    def fromSplitLists(cls, trn_x = None, trn_y = None, val_x = None, val_y = None, test_x = None, test_y = None, trn_folder = '.', val_folder = '.', test_folder='.', suffix='.jpg', bs =2, imSize = (224,224), transforms = []):
        trn_ds, val_ds, test_ds = None, None, None;
        if trn_x:
            trn_ds = cls.DS(trn_x, trn_y, trn_folder, imSize, suffix, transforms)
        if val_x:
            val_ds = cls.DS(val_x, val_y, val_folder, imSize, suffix)
        if test_x:
            test_ds = cls.DS(test_x, test_y, test_folder, imSize, suffix)
        return cls(trn_ds, val_ds, test_ds, bs)


    @classmethod
    def fromSingleList(cls, x,y, split_ratio = [0.8, 0.2], folder = '.', imSize = (224,224), suffix='.jpg', bs=2, transforms = []):
        trn_ds, val_ds, test_ds = None, None, None
        split_indices = [ int(split_ratio[0] * len(x)), int((split_ratio[0]+split_ratio[1]) * len(x)) ]
        return  cls.fromSplitLists( x[:split_indices[0]], y[: split_indices[0]], x[split_indices[0] : split_indices[1]], y[ split_indices[0] : split_indices[1] ],
                x[split_indices[1] :], y[split_indices[1]:], folder, folder, folder, suffix, bs, imSize, transforms)


class DetectionDataset(ClassDataset):

    def preprocY(self, index, transformVals):
        y = tensorY(self.y[index], self.imSize, self.filePath(index),
                transformVals, max([len(row) for row in self.y]) )
        return y


class DetectionDataPack(ClassDataPack):
    DS = DetectionDataset


__all__ = ['ClassDataset', 'ClassDataPack', 'DetectionDataset', 'DetectionDataPack']
