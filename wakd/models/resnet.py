import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models


class resnet(nn.Module):
    __fix_layers = {
        'conv5':7,
        'conv4':6,
        'conv3':5,
        'conv2':4,
        'full':0
    }


    def __init__(self, depth, train_layers='full', model_path=None):
        super(resnet, self).__init__()
        if depth == 34:
            self.model = models.resnet34(pretrained=False)
            self.feature_dim = 512
        elif depth == 50:
            self.model = models.resnet50(pretrained=False)
            self.feature_dim = 2048
        elif depth == 101:
            self.model = models.resnet101(pretrained=False)
            self.feature_dim = 2048
        elif depth == 152:
            self.model = models.resnet152(pretrained=False)
            self.feature_dim = 2048
        self.modelPath = model_path
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveMaxPool2d(1)
        # self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self._init_params()

        layers = [self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
                    self.model.layer1, self.model.layer2, self.model.layer3]
        for l in layers[:resnet.__fix_layers[train_layers]]:
            for p in l.parameters():
                p.requires_grad = False
	

    def _init_params(self):
        # optional load pretrained weights
        if (self.modelPath is not None):
            self.model.load_state_dict(torch.load(self.modelPath))


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # x = self.upSample(x)
        
        # pool_x = self.model.avgpool(x)
        pool_x = self.gap(x)
        pool_x = pool_x.view(pool_x.size(0), -1)
        return pool_x, x


def resnet50(**kwargs):
    return resnet(depth=50, **kwargs)


def resnet34(**kwargs):
    return resnet(depth=34, **kwargs)
    

def resnet101(**kwargs):
    return resnet(depth=101, **kwargs)
    
    
def resnet152(**kwargs):
    return resnet(depth=152, **kwargs)

