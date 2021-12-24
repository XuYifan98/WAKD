from __future__ import absolute_import
from .vgg import *
from .nextvlad import NextVLAD
from .netvlad import *
from .resnet import *
from .mobilenetv3 import *
from .iresgroup import *

__factory = {
    'vgg16': vgg16,
    'vgg19': vgg19,
    'mobilenetv3_large': mobilenetv3_large,
    'mobilenetv3_small': mobilenetv3_small,
    'resnet50': resnet50,
    'resnet152': resnet152,
    'iresgroup101': iresgroup101,
    'iresgroup152': iresgroup152,
    'nextvlad': NextVLAD,
    'netvlad': NetVLAD,
    'embednet': EmbedNet,
    'embednetpca': EmbedNetPCA,
    'embedregionnet': EmbedRegionNet,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)