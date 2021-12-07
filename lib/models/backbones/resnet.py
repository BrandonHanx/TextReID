from collections import namedtuple

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    # original padding is 1; original dilation is 1
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # original padding is 1; original dilation is 1
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        model_arch,
        res5_stride=2,
        res5_dilation=1,
        pretrained=True,
    ):
        super().__init__()
        block = model_arch.block
        layers = model_arch.stage

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=res5_stride, dilation=res5_dilation
        )

        if pretrained is None:
            self.load_state_dict(remove_fc(model_zoo.load_url(model_arch.url)))
        else:
            self.load_state_dict(torch.load(pretrained))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = 512 * block.expansion

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def remove_fc(state_dict):
    """Remove the fc layer parameters from state_dict."""
    for key in list(state_dict.keys()):
        if key.startswith("fc."):
            del state_dict[key]
    return state_dict


resnet = namedtuple("resnet", ["block", "stage", "url"])
model_archs = {}
model_archs["resnet18"] = resnet(
    BasicBlock,
    [2, 2, 2, 2],
    "https://download.pytorch.org/models/resnet18-5c106cde.pth",
)
model_archs["resnet34"] = resnet(
    BasicBlock,
    [3, 4, 6, 3],
    "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
)
model_archs["resnet50"] = resnet(
    Bottleneck,
    [3, 4, 6, 3],
    "https://download.pytorch.org/models/resnet50-19c8e357.pth",
)
model_archs["resnet101"] = resnet(
    Bottleneck,
    [3, 4, 23, 3],
    "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
)
model_archs["resnet152"] = resnet(
    Bottleneck,
    [3, 8, 36, 3],
    "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
)


def build_resnet(cfg):
    arch = cfg.MODEL.VISUAL_MODEL
    res5_stride = cfg.MODEL.RESNET.RES5_STRIDE
    res5_dilation = cfg.MODEL.RESNET.RES5_DILATION
    pretrained = cfg.MODEL.RESNET.PRETRAINED

    model_arch = model_archs[arch]
    model = ResNet(
        model_arch,
        res5_stride,
        res5_dilation,
        pretrained=pretrained,
    )

    if cfg.MODEL.FREEZE:
        for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
