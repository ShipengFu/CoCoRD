from .resnet import resnet20, resnet32, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg8_bn, vgg11_bn, vgg13_bn, vgg19_bn
from .ResNet import ResNet18, ResNet34, ResNet50, ResNet101
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2

model_dict = {
    'vgg8': vgg8_bn,
    'vgg13': vgg13_bn,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'vgg11': vgg11_bn,
    'vgg19': vgg19_bn,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
}

