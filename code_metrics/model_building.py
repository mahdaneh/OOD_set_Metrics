


import torch.nn.functional as F
import torch.nn.init as init

import torch.nn as nn
import math
"""
UTILS CLASSES
"""


def in_feature_size (ft_extrctor_p, img_size):
    w = img_size
    for lyr_name, lyr_func in ft_extrctor_p.items():
        for oprt in lyr_func.keys():
            prmtrs = lyr_func[oprt]

            if oprt == 'conv':
                num_filters = prmtrs[1]
                w = int ((w - prmtrs[2] + 2*prmtrs[4]) /prmtrs[3])+1

            elif oprt =='maxpool':
                w = int((w - prmtrs[0] + 2 * prmtrs[2]) / prmtrs[1])+1
        print (w,w)
    return w*w*num_filters


"""
UTILS CLASSES
"""
#
#
# class feature_extractor():
#     def __init__(self, ft_extrctr_p):
#         self.ft_extrctr_p = ft_extrctr_p
#
#     def construct(self):
#         layers = []
#         for lyr_name, lyr_func in self.ft_extrctr_p.items():
#             for oprt in lyr_func.keys():
#                 prmtrs = lyr_func[oprt]
#                 if oprt == 'conv':
#                     # in_channels, out_channels, kernel_size, stride = 1, padding = 0
#                     layers += [
#                         nn.Conv2d(in_channels=prmtrs[0], out_channels=prmtrs[1], kernel_size=prmtrs[2],
#                                   stride=prmtrs[3],
#                                   padding=prmtrs[4])]
#
#                 if oprt == 'relu':
#                     layers += [(nn.ReLU(True))]
#                 if oprt=='elu':
#                     layers += [(nn.ELU(True))]
#                 if oprt == 'dropout':
#                     layers += [nn.Dropout(p=prmtrs[0])]
#                 if oprt == 'maxpool':
#                     layers += [nn.MaxPool2d(kernel_size=prmtrs[0], stride=prmtrs[1], padding=prmtrs[2])]
#         return nn.Sequential(*layers)


class classifier(nn.Module):
    def __init__(self, FE_p, classifier_p):
        super(classifier, self).__init__()


        layers = self._construct(FE_p)
        self.feature_extrct = nn.Sequential(*layers)


        layers = self._construct(classifier_p)
        self.h = nn.Sequential(*layers)

    def _construct(self,all_layers_name):

        layers = []
        for lyr_name in all_layers_name:
            lyr_func = all_layers_name[lyr_name]
            print (lyr_name, lyr_func)
            for oprt in lyr_func.keys():
                prmtrs = lyr_func[oprt]
                if oprt == 'conv':
                    # in_channels, out_channels, kernel_size, stride = 1, padding = 0
                    layers += [
                        nn.Conv2d(in_channels=prmtrs[0], out_channels=prmtrs[1], kernel_size=prmtrs[2],
                                  stride=prmtrs[3],
                                  padding=prmtrs[4])]

                if oprt == 'relu':
                    layers += [(nn.ReLU(True))]
                if oprt=='elu':
                    layers += [(nn.ELU(True))]
                if oprt == 'dropout':
                    layers += [nn.Dropout(p=prmtrs[0])]
                if oprt == 'maxpool':
                    layers += [nn.MaxPool2d(kernel_size=prmtrs[0], stride=prmtrs[1], padding=prmtrs[2])]

                if oprt == 'fc':
                    layers += [nn.Linear(prmtrs[0], prmtrs[1])]

                if oprt == 'dropout':
                    layers += [nn.Dropout(p=prmtrs[0])]


        return  layers
    def forward(self, x):


        x = self.feature_extrct(x)

        x = x.view(x.shape[0],-1)

        x = self.h(x)
        return x

    def feature_list(self, x):
        out_list = []
        x = self.feature_extrct(x)
        x = x.view(x.shape[0], -1)
        out_list += [x]
        x = self.h(x)
        out_list += [x]
        y =None
        return y, out_list



def MTL_network(chan, imgsize,num_class):
        ft_extrctor_prp = {'layer1_f': {'conv': [chan, 32, 5, 1, 2], 'relu': [], 'maxpool': [3, 2, 0]},
                            'layer2_f': {'conv': [32, 64, 5, 1, 2], 'relu': [], 'maxpool': [3, 2, 0]},
                                               }

        in_features =  (in_feature_size(ft_extrctor_prp, imgsize))
        print (in_features)

        hypoth_prp = {'layer3': {'fc': [in_features, 128], 'elu':[]},
                      'layer4': {'fc': [128, num_class]}}

        model = classifier(ft_extrctor_prp, hypoth_prp)
        return model

class cuda_conv_18 (nn.Module):
    def __init__(self, num_class, num_fill =[32,32,64]):
        super(cuda_conv_18, self).__init__()
        self.conv1 = nn.Conv2d(1, num_fill[0], 5, 1)
        self.conv2 = nn.Conv2d(num_fill[0], num_fill[1], 5, 1)
        self.conv3 = nn.Conv2d(num_fill[1], num_fill[2], 5, 1)
        self.fc1 = nn.Linear(500, num_class)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 3, 2)
        print (x.shape)
        x = x.view(-1, 5 * 5 * 50)

        x = self.fc1(F.dropout(x))

        return x


class cuda_conv(nn.Module):
    def __init__(self,num_class):
        super(cuda_conv, self).__init__()
        self.blocks = []
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.blocks +=[self.conv1]
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.blocks += [self.conv2]
        self.fc1 = nn.Linear(5* 5 * 50, 500)
        self.blocks += [self.fc1]
        self.fc2 = nn.Linear(500, num_class)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 5*5*50)

        x = F.relu(self.fc1(F.dropout(x)))

        x = self.fc2(x)

        return x


    def feature_list(self, x):
        out_list = []

        x = F.relu(self.conv1(x))
        out_list+=[x]
        x = F.max_pool2d(x, 2, 2)
        out_list += [x]
        x = F.relu(self.conv2(x))
        out_list += [x]
        x = F.max_pool2d(x, 2, 2)
        out_list += [x]
        x = x.view(-1, 5*5*50)
        out_list += [x]
        x = F.relu(self.fc1(F.dropout(x)))
        out_list += [x]
        y =None
        return y, out_list



'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




def resnet20(num_class):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_class)


def resnet32(num_class):
    return ResNet(BasicBlock, [5, 5, 5],num_classes=num_class)


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(num_class):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_class)


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

#
# def test(net):
#     import numpy as np
#     total_params = 0
#
#     for x in filter(lambda p: p.requires_grad, net.parameters()):
#         total_params += np.prod(x.data.numpy().shape)
#     print("Total number of params", total_params)
#     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))
#
#
#
# pytorch,numpy,list are muutable.

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class MLP(nn.Module):
    def __init__(self, num_filters =None, num_class=1):
        super(MLP, self).__init__()
        self.layers =[]
        for i, f in enumerate(num_filters):
            p=0 if i==0 else 0.1
            self.layers.append([nn.Dropout(p),
            nn.Linear(f[0], f[1] ),
            nn.ReLU(True)])

        self.classifier = nn.Sequential(nn.Linear(f[1], num_class), nn.Sigmoid())

    def forward(self,x):
        for l in self.layers:
            x = nn.Sequential(*l)(x)

        x = self.classifier(x)
        return  x
    def feature_list(self,x):
        output_layer  =[ ]
        for l in self.layers:
            x = nn.Sequential(*l)(x)
            output_layer.append(x)
        y = self.classifier(x)
        return  y , output_layer





class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes, classifier_spc=None):
        super(VGG, self).__init__()
        self.features = features[0]
        self.block = features[1]
        self.num_classes = num_classes

        if classifier_spc is None:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True)
            )
            out_fil = 512
        else:
            temp = []
            in_fil = 512
            for filters in classifier_spc:
                temp+= [nn.Dropout(),
                nn.Linear(in_fil, filters ),
                nn.ReLU(True)]

                in_fil = filters

            self.classifier = nn.Sequential(*temp)
            out_fil = in_fil

        self.output = nn.Linear(out_fil, self.num_classes)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        # x = nn.Sequential(*self.features)(x)
        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        x = self.output(x)



        return x
        # function to extact the multiple features

    def feature_list(self, x):
        out_list = []

        for layer in self.block:

            x = nn.Sequential(*layer)(x)
            out_list.append(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        out_list.append(x)
        y = self.output(x)
        return y, out_list

    def intermediate_forward(self, x, layer_index):
        for j in range(layer_index ):
            x = nn.Sequential(*self.block[j])(x)
        return x

    def penultimate_forward(self, x):
        for layer in self.block:
            x = nn.Sequential(*layer)(x)


        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        penultimate = x
        x = self.output(x)
        return x, penultimate


def make_layers(cfg, batch_norm=False):
    layers = []
    sub_block = []
    blocks= []

    in_channels = 3
    for v in cfg:
        if v == 'M':
            mpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
            layers += [mpool2d]
            sub_block +=[mpool2d]
            blocks.append(sub_block)
            sub_block = []
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            batchnorm2d = nn.BatchNorm2d(v)
            relu2d = nn.ReLU(inplace=True)

            if batch_norm:
                layers += [conv2d, batchnorm2d, relu2d]
                sub_block+= [conv2d, batchnorm2d, relu2d]
            else:
                layers += [conv2d, relu2d]
                # block.append([conv2d, relu2d])
                sub_block+=[conv2d, relu2d]
            in_channels = v
    return (nn.Sequential(*layers), blocks)



cfg = {
    '-A':[64, 'M', 128, 'M',  128, 'M',128, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256,  'M',256, 256, 'M', 512,  512,'M',512, 512, 'M',
          512, 512, 'M'],
}

def     mini_vgg(num_class):
    return  VGG(make_layers(cfg['-A']), num_classes= num_class, classifier_spc = [128])
def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn(num_class):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_class)


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16(num_class):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), num_classes=num_class)


def vgg16_bn(num_class):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_class)

def vgg19(num_class):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), num_classes = num_class)


def vgg19_bn(num_class):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes = num_class)
