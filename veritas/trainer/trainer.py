import math
import torch
from torch import optim, nn
import torch.nn.functional as F
from ipywidgets import IntProgress,Label,Layout,HBox
from IPython.display import display
from ..utils.numeric import avg, to_np
from ..utils.model import groupParams, getParamsArr


__all__ = ['Trainer', 'GanTrainer']


class Trainer:
    def __init__(self, m , md , optFn= optim.SGD, lossFn = F.binary_cross_entropy_with_logits,
            metrics = []):
        self.model = m
        self.md = md
        self.optFn = optFn
        self.lossFn = lossFn


    def train(self, cycle =1, maxLR = 1e-2, baseLR = None, epochPerCycle = 1, lrFactors=[1] ):
        progressBar = ProgressBar()
        baseLR = baseLR if baseLR else maxLR/10
        cumTrnLoss, cumValLoss = None, None
        for currentEpoch in range(epochPerCycle * cycle ):
            if hasattr(self.md, 'trn_dl'):
                cumTrnLoss = self.trainEpoch(progressBar, 'trn', currentEpoch, epochPerCycle,
                        cycle, maxLR, baseLR, lrFactors)
            if hasattr(self.md, 'val_dl'):
                cumValLoss = self.trainEpoch(progressBar, 'val', currentEpoch, epochPerCycle, cycle)
            print('epoch', currentEpoch+1, '/', epochPerCycle * cycle, 'trn loss:', cumTrnLoss, 'val loss:', cumValLoss)
        return


    def trainEpoch(self, progressBar, dsType = 'trn', currentEpoch = None,
            epochPerCycle = None, cycle = None, maxLR = None, baseLR = None, lrFactors = [1]):
        self.model.train() if dsType == 'trn' else self.model.eval()
        dl = self.md.trn_dl if dsType == 'trn' else self.md.val_dl
        ds = self.md.trn_ds if dsType == 'trn' else self.md.val_ds
        batchLosses = []
        for currentIter , (x, y) in enumerate(dl ):
            x, y = x.cuda(), y.cuda()
            loss = self.trainBatch(x, y, dsType, maxLR, baseLR, lrFactors, cycle, epochPerCycle, currentEpoch, currentIter, math.ceil( ds.__len__() / self.md.bs ) )
            batchLosses.append(loss)
            progressBar.updateStats(currentIter+1, math.ceil( ds.__len__() / self.md.bs ), currentEpoch+1, epochPerCycle * cycle, loss)
        return avg(batchLosses)


    def trainBatch(self, x, y, dsType, maxLR, baseLR, lrFactors, cycle, epochPerCycle, currentEpoch, currentIter, batchCount):
        out = self.model(x)
        loss = self.lossFn(out, y)
        if self.model.training:
            lr = annealLR(maxLR, baseLR, cycle, epochPerCycle, currentEpoch, currentIter, batchCount)
            #opt = self.optFn(self.model.parameters(),lr)
            opts = getOptimizers(self.model, self.optFn, lr,lrFactors)
            for opt in opts: opt.zero_grad()
            loss.backward()
            for opt in opts: opt.step()

        return to_np(loss)


    def predict(self):
        self.model.eval()
        if(self.md.test_dl):
            dl = self.md.test_dl
            results = []
            ys =[]
            for x,y in dl:
                x, y = x.cuda(), y.cuda()
                results.append( self.model(x).detach() )
                ys.append(y.detach())
            return torch.cat(results,dim=0), torch.cat(ys, dim =0)

    def freeze_to(self, layer = -1):
        params = list(self.model.parameters())
        for param in params[ :layer * len(params)//10 ]:
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

class GanTrainer(Trainer):
    def __init__(self, genM,discM, md, k, optFn, metrics=[] ):
        super().__init__(genM,md, optFn, F.cross_entropy, metrics);
        self.discModel = discM
        self.k = k

    def trainBatch(self, x, y, dsType, maxLR, baseLR, lrFactors, cycle, epochPerCycle, currentEpoch, currentIter, batchCount):
        #train the discriminator
        lr = annealLR(maxLR, baseLR, cycle, epochPerCycle, currentEpoch, currentIter, batchCount)
        for i in range(self.k):
            opt = self.optFn(self.discModel.parameters(), lr)
            opt.zero_grad()
            lossD1 = self.lossFn(self.discModel(self.model(x).detach()), torch.zeros(x.shape[0]).long().cuda())
            lossD1.backward()
            lossD2 = self.lossFn(self.discModel(y), torch.ones(x.shape[0]).long().cuda())
            lossD2.backward()
            opt.step()
        opt = self.optFn(self.model.parameters(), lr)
        opt.zero_grad()
        lossG = self.lossFn(self.discModel(self.model(x)), torch.ones(x.shape[0]).long().cuda())
        lossG.backward()
        opt.step()
        return to_np(F.l1_loss(self.model(x), y))

class ProgressBar:
    def __init__(self):
        self.t1 = Label('', layout = Layout(width='10%',height='10%'))
        self.f = IntProgress(min=0, max = 100)
        self.t2 = Label('', layout = Layout(width='10%',height='10%'))
        self.t3 = Label('loss:- acc:-',layout= Layout(width='30%',height='10%'))
        self.hb = HBox([self.t1,self.f,self.t2,self.t3])
        display(self.hb)
        return


    def showProgressBar(self):
        display(self.hb)
        return


    def updateStats(self,i, leny, epoch, epochs, loss):
        self.f.value = i/leny*100
        self.t1.value = str(epoch)+' / '+str(epochs)
        self.t2.value = str(i)+' / '+str(leny)
        self.t3.value = 'loss: {0:.4f}'.format(loss)
        return

def optimize(model, loss, lr, lrFactors, optFn):
    opts = getOptimizers(model, optFn, lr, lrFactors)
    for opt in opts: opt.zero_grad()
    loss.backward()
    for opt in opts: opt.step()

def annealFactor(cycle,epochPerCycle, currentEpoch, currentIter, batchCount):
    effCurrentIter = currentEpoch% epochPerCycle * batchCount + currentIter
    medianIter = epochPerCycle * batchCount //2
    factor = 1 - abs(medianIter - effCurrentIter)/medianIter
    return factor


def annealLR(maxLR, baseLR, cycle, epochPerCycle, batchCount, currentEpoch, currentIter):
    return annealFactor(cycle,epochPerCycle, batchCount, currentEpoch, currentIter) * (maxLR - baseLR) + baseLR


def getOptimizers(m, optFn, lr, lrFactors = [1]):
    #return [optFn(m.parameters(), lr)]
    if not lrFactors:
        lrFactors = [1]
    modelParamsArr = getParamsArr(m)
    paramsGroups = groupParams(m, lrFactors)
    opts = sum([[optFn([p], lr*lr_factor) for p in pG]
            for pG,lr_factor in zip(paramsGroups, lrFactors)], [])
    return opts
