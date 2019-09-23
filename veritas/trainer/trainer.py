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
    self.model = m;
    self.md = md;
    self.optFn = optFn;
    self.lossFn = lossFn;

  def fetchLRs(self, epoch, maxLR, baseLR):
    batchCount = epoch * len(self.md.trn_dl);
    return [(1-(abs(i-batchCount//2)/(batchCount//2)))*(maxLR - baseLR) +
        baseLR for i in range(batchCount) ];

  def train(self, epoch, maxLR = 1e-2, baseLR = None ):
    progressBar = ProgressBar()
    if not baseLR : baseLR = maxLR;
    cumTrnLoss, cumValLoss = None, None
    lrs = self.fetchLRs(epoch, maxLR, baseLR);
    for epochIdx in range(epoch):
      if hasattr(self.md, 'trn_dl'):
        self.model.train();
        cumTrnLoss = self.trainEpoch(progressBar, self.md.trn_dl, epochIdx, epoch, lrs);
      if hasattr(self.md, 'val_dl'):
        self.model.eval();
        cumValLoss = self.trainEpoch(progressBar, self.md.val_dl, epochIdx, epoch);
        print('epoch', epochIdx+1, '/', epoch, 'trn loss:', cumTrnLoss,
            'val loss:', cumValLoss);
    return cumTrnLoss, cumValLoss;

  def trainEpoch(self, progressBar, dl, epochIdx, epoch, lrs = None):
    batchLosses = [];
    for batchIdx , (x, y) in enumerate(dl ):
      lr = lrs[ epochIdx*len(dl)+batchIdx] if lrs else None
      x, y = x.cuda(), y.cuda();
      loss = self.trainBatch(x, y, lr)
      batchLosses.append(loss)
      progressBar.updateStats(batchIdx+1, len(dl), epochIdx+1, epoch, loss)
    return avg(batchLosses)


  def trainBatch(self, x, y, lr, lrFactors = [1]):
    out = self.model(x)
    loss = self.lossFn(out, y)
    if self.model.training:
      opts = getOptimizers(self.model, self.optFn, lr,lrFactors)
      for opt in opts: opt.zero_grad()
      loss.backward()
      for opt in opts: opt.step()
    return to_np(loss)


  def predict(self):
    self.model.eval()
    if(self.md.test_dl):
      results = []
      ys =[]
      for x,y in self.md.test_dl:
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
  def __init__(self, genM,discM, md, wgan = False, k=1, c=0.01, optFn=optim.RMSprop, metrics=[] ):
    super().__init__(genM,md, optFn, F.cross_entropy, metrics);
    self.discModel = discM
    self.k = k if wgan else 1
    self.c = c if wgan else 0

    def trainBatch(self, x, y, lr, lrFactors):
      #train the discriminator
      lossD = 0
      for i in range(self.k):
        opt = self.optFn(self.discModel.parameters(), lr)
        opt.zero_grad()
        if not self.wgan:
          lossD1 = self.lossFn(self.discModel(self.model(x).detach()), torch.zeros(x.shape[0]).long().cuda())
          lossD1.backward()
          lossD2 = self.lossFn(self.discModel(y), torch.ones(x.shape[0]).long().cuda())
          lossD2.backward()
          opt.step()
          lossD += lossD1 + lossD2
        else:
          lossD1 =  torch.mean(self.discModel(self.model(y)))
          lossD1.backward()
          lossD2 = - torch.mean(self.discModel( self.model(x).detach() ))
          lossD2.backward()
          opt.step()
          lossD += lossD1 + lossD2
          #print(to_np(lossD1), to_np(lossD2))
          for p in self.discModel.parameters():
            if self.c : p.data.clamp_(-self.c, self.c )
        opt = self.optFn(self.model.parameters(), lr)
        opt.zero_grad()
        lossG = None
        if not self.wgan:
          lossG = self.lossFn(self.discModel(self.model(x)), torch.ones(x.shape[0]).long().cuda())
        else:
          lossG = torch.mean(self.discModel(self.model(x)))
        lossG.backward()
        opt.step()
        #return to_np(F.l1_loss(self.model(x), y))
        return to_np(lossG)

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

def annealFactor(cycle,epochPerCycle, epochIdx, currentIter, batchCount):
  effCurrentIter = epochIdx% epochPerCycle * batchCount + currentIter
  medianIter = epochPerCycle * batchCount //2
  factor = 1 - abs(medianIter - effCurrentIter)/medianIter
  return factor


def annealLR(maxLR, baseLR, cycle, epochPerCycle, batchCount, epochIdx, currentIter):
  return annealFactor(cycle,epochPerCycle, batchCount, epochIdx, currentIter) * (maxLR - baseLR) + baseLR


def getOptimizers(m, optFn, lr, lrFactors = [1]):
  #return [optFn(m.parameters(), lr)]
  if not lrFactors:
    lrFactors = [1]
  modelParamsArr = getParamsArr(m)
  paramsGroups = groupParams(m, lrFactors)
  opts = sum([[optFn([p], lr*lr_factor) for p in pG]
  for pG,lr_factor in zip(paramsGroups, lrFactors)], [])
  return opts
