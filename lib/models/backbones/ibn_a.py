import math

import torch
import torch.nn as nn
import torch.nn.functional as F

model_urls = {
    "resnet18_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth",
    "resnet34_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth",
    "resnet50_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth",
    "resnet101_ibn_a": "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth",
}


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        if ibn == "a":
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == "b" else None
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
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == "a":
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == "b" else None
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
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim, embed_dim, num_heads, patch_size=1, output_dim=None
    ):
        super().__init__()
        self.proj_conv = None
        if patch_size > 1:
            self.proj_conv = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
        self.positional_embedding = nn.Parameter(
            torch.randn(
                (spacial_dim[0] // patch_size) * (spacial_dim[1] // patch_size) + 1,
                embed_dim,
            )
            / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        if self.proj_conv is not None:
            x = self.proj_conv(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ResNet_IBN(nn.Module):
    def __init__(
        self,
        last_stride,
        block,
        layers,
        ibn_cfg=("a", "a", "a", None),
        attn_pool=False,
    ):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if ibn_cfg[0] == "b":
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride, ibn=ibn_cfg[3]
        )

        self.attn_pool = None
        self.out_channels = 512 * block.expansion
        if attn_pool:
            output_dim = 1024  # FIXME: need flexible
            input_dim = (384, 128)
            embed_dim = self.out_channels
            self.out_channels = output_dim
            down_ratio = 16 if last_stride == 1 else 32
            num_heads = 32
            spacial_dim = (input_dim[0] // down_ratio, input_dim[1] // down_ratio)
            self.attn_pool = AttentionPool2d(
                spacial_dim, embed_dim, num_heads, 1, output_dim
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
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
        layers.append(
            block(
                self.inplanes, planes, None if ibn == "b" else ibn, stride, downsample
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    None if (ibn == "b" and i < blocks - 1) else ibn,
                )
            )

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

        if self.attn_pool is not None:
            x = self.attn_pool(x)
        return x


def remove_fc(state_dict):
    """Remove the fc layer parameters from state_dict."""
    for key in list(state_dict.keys()):
        if key.startswith("fc.") or key.startswith("avgpool."):
            del state_dict[key]
    return state_dict


def resnet18_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-18-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        last_stride=last_stride,
        block=BasicBlock_IBN,
        layers=[2, 2, 2, 2],
        ibn_cfg=("a", "a", "a", None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            remove_fc(torch.hub.load_state_dict_from_url(model_urls["resnet18_ibn_a"])),
            strict=False,
        )
    return model


def resnet34_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-34-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        last_stride=last_stride,
        block=BasicBlock_IBN,
        layers=[3, 4, 6, 3],
        ibn_cfg=("a", "a", "a", None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            remove_fc(torch.hub.load_state_dict_from_url(model_urls["resnet34_ibn_a"])),
            strict=False,
        )
    return model


def resnet50_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        last_stride=last_stride,
        block=Bottleneck_IBN,
        layers=[3, 4, 6, 3],
        ibn_cfg=("a", "a", "a", None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            remove_fc(torch.hub.load_state_dict_from_url(model_urls["resnet50_ibn_a"])),
            strict=False,
        )
    return model


def resnet101_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(
        last_stride=last_stride,
        block=Bottleneck_IBN,
        layers=[3, 4, 23, 3],
        ibn_cfg=("a", "a", "a", None),
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            remove_fc(
                torch.hub.load_state_dict_from_url(model_urls["resnet101_ibn_a"])
            ),
            strict=False,
        )
    return model


def build_ibn_a(cfg):
    arch = cfg.MODEL.VISUAL_MODEL
    res5_stride = cfg.MODEL.RESNET.RES5_STRIDE
    attn_pool = cfg.MODEL.RESNET.ATTN_POOL

    if arch == "resnet50_ibn_a":
        model = resnet50_ibn_a(res5_stride, pretrained=True, attn_pool=attn_pool)
    elif arch == "resnet101_ibn_a":
        model = resnet101_ibn_a(res5_stride, pretrained=True, attn_pool=attn_pool)
    elif arch == "resnet34_ibn_a":
        model = resnet34_ibn_a(res5_stride, pretrained=True, attn_pool=attn_pool)

    return model
