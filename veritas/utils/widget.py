from ipywidgets import IntProgress,Label,Layout,HBox
from IPython.display import display

class ProgressBar:
    def __init__(self):
        self.t1 = Label('', layout = Layout(width='10%',height='10%'))
        self.f = IntProgress(min=0, max = 100)
        self.t2 = Label('', layout = Layout(width='10%',height='10%'))
        self.t3 = Label('loss:- acc:-',layout= Layout(width='30%',height='10%'))
        self.hb = HBox([self.t1,self.f,self.t2,self.t3])
    def showProgressBar(self):
        display(self.hb)

    def updateProgress(self,i,leny):
        self.f.value = i/leny*100
        self.t2.value = str(i)+' / '+str(leny)

    def updateEpochCount(self,epoch,epochs):
        self.t1.value = str(epoch)+' / '+str(epochs)

    def updateStats(self,loss,acc):
        self.t3.value = 'loss: {0:.4f}, acc: {1:.4f}'.format(loss,acc)


