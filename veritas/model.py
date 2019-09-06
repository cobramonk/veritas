import torch
from torch import nn
from torchvision import models

class classifier(nn.Module):
    def __init__(self, cats, dps, model = 'resnet34', custom_head = None):
        super().__init__()
        model = getattr(models,model);
        self.body = nn.Sequential( *(list(model(pretrained = True).children())[:-2]))
        self.head = nn.sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d( list(self.body.parameters())[-1].shape[0] ),
            nn.Dropout(0.5),
            nn.Linear( list(self.body.parameters())[-1].shape[0] ,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512,cats)
        )
        if custom_head:
            self.head = custom_head

    def forward(self, x):
        return self.head(self.body(x))

class ImgClas(nn.Module):
    def __init__(self,cats,dps,model = 'resnet34', custom_head = None):
        super().__init__()
        nodes = {'resnet34':1024,'resnet101':4096,'resnet50':2048}
        nds = nodes[model]
        model = getattr(models,model);
        body = model(pretrained=True)
        body = list(body.children())[:-2]
        self.feet = nn.Sequential(*body[0:6])
        self.torso = nn.Sequential(*body[6:])
        head = nn.Sequential(
                AdaptiveConcatPool2d(),
                Flatten(),
                nn.BatchNorm1d(nds),
                nn.Dropout(0.25),
                nn.Linear(nds,512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512,cats)
                )
        if custom_head:
            self.head = custom_head
        else:
            self.head = head

    def forward(self,x):
        x = self.torso(self.feet(x))
        return self.head(x)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = x.view(x.size(0),-1)
        return x

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
    def forward(self,x):
        return torch.cat((self.ap(x),self.mp(x)),1)

