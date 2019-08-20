import numpy as np
import torch
from torchvision.transforms import functional as TF
from PIL import Image

def pilImage(fpath, imSize, transformVals):
    img = TF.resize( Image.open( fpath ), imSize )
    img = transformImage(img,transformVals)
    return img

def tensorImage(fpath, imSize, transformVals):
    img = pilImage(fpath, imSize, transformVals)
    return TF.to_tensor(img)[:3].contiguous()

def transformImage(img, transformVals):
    if transformVals[0] == 1:
        img = TF.hflip(img)
    img = TF.affine(img, transformVals[1], transformVals[2], transformVals[3],0)
    return img

def transformValGen(transforms):
    transformVals = [0,0,(0,0),1]
    if 'hflip' in transforms:
        transformVals[0] = np.random.randint(0,2)
    if 'rotate' in transforms:
        transformVals[1] = np.random.randint(-15,15)
    if 'translate' in transforms:
        transformVals[2] = (np.random.randint(0,15), np.random.randint(0,15))
    if 'zoom' in transforms:
        transformVals[3] = np.random.rand() * 0.1 + 1
    return transformVals

def tensorY(y, imSize, imFpath, transformVals, maxObjCount):
    y = np.float32(y)
    y = resizeY(y, imSize, Image.open(imFpath).size)
    y = padY(y, maxObjCount)
    y = transformY(y, imSize, transformVals)
    return torch.tensor(y).float()

def transformY(y, imSize, transformVals ):
    if transformVals[2] != (0,0):
        y = translateY(y, transformVals[2], imSize)
    if transformVals[3] != 0:
        y = zoomY(y, transformVals[3], imSize)
    return y


def resizeY(y, imSize, imOrgSize):
    y[:,:4] = np.float32([ [ y[i,j] * imSize[j%2]// imOrgSize[j%2] for j in range(4) ]for i in range(len(y)) ])
    return y

def padY(y, maxObjCount):
    if maxObjCount == len(y): return y
    return np.concatenate((y, np.float32([[0,0,0,0,0] for l in range(maxObjCount - len(y))]) ))

def translateY(y, trans, imSize):
    y[:,:4] =  np.float32([ [y[i, j]+trans[j%2] for j in range(4) ] for i in range( len(y) )])
    return y

def zoomY(y, zoom , imSize):
    y[:, :4] = _shiftToCenter(y[:,:4], imSize)
    y[:, :4] = np.float32([ [e *zoom for e in box[:4] ] for box in y])
    y[:, :4] = _shiftToTopleft(y[:, :4], imSize)
    return y

def _shiftToCenter(y, imSize):
    return y - np.float32([ [ imSize[0]//2, imSize[1]//2 ] *2 for i in range(len(y)) ])

def _shiftToTopleft(y, imSize):
    return y + np.float32([ [ imSize[0]//2, imSize[1]//2 ] *2 for i in range(len(y)) ])


def get_grids( im_size, grid_dims ):
    grids = []
    for grid_dim in grid_dims:
        curr_dim_grids = [ [j/grid_dim, i/grid_dim , (j+1)/grid_dim, (i+1)/grid_dims]
                for i in range(grid_dim) for j in range(grid_dim)]
    return grids

def get_anchors(im_size, grid_dims, aspect_ratios, zoom, k):
    pass
