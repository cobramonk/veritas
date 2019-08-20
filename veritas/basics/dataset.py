from torch.utils import data as dd
import numpy as np

__all__ = ['BasicDataset', 'BasicDataPack']

class BasicDataset(dd.Dataset):
    def __init__(self, x,y):
        #x is the input and y is the target
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self,index):
        return np.float32(self.x[index]), np.float32(self.y[index])


class BasicDataPack:
    def __init__(self,trn_ds,val_ds=None,test_ds=None,bs=1):
        self.trn_ds = trn_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.bs = bs
        self.create_dl()

    def create_dl(self):
        if self.trn_ds:
            self.trn_dl = dd.DataLoader(self.trn_ds,self.bs)
        if self.val_ds:
            self.val_dl = dd.DataLoader(self.val_ds,self.bs)
        if self.test_ds:
            self.test_dl = dd.DataLoader(self.test_ds, self.bs)

    @classmethod
    def fromSplitLists(cls,trn_x = None, trn_y = None, val_x = None, val_y = None, test_x = None, test_y = None, bs = 2):
        trn_ds, val_ds, test_ds = None, None, None
        if trn_x:
            trn_ds = BasicDataset(trn_x, trn_y)
        if val_x:
            val_ds = BasicDataset(val_x, val_y)
        if test_x:
            test_ds = BasicDataset(test_x, test_y)
        return cls(trn_ds, val_ds, test_ds, bs)

    @classmethod
    def fromSingleList(cls, x, y, split_ratio = [0.8, 0.2, 0], bs = 2):
        trn_ds, val_ds, test_ds = None, None, None
        split_indices = [ int(split_ratio[0] * len(x)), int((split_ratio[0]+split_ratio[1]) * len(x)) ]
        return  cls.fromSplitLists(
                x[:split_indices[0]], y[: split_indices[0]],
                x[split_indices[0] : split_indices[1]], y[ split_indices[0] : split_indices[1] ],
                x[split_indices[1] :], y[split_indices[1]:],
                bs
                );


