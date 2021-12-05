from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ResNet(nn.Module):
    def __init__(
        self,
        model_arch,
        res5_stride=2,
        res5_dilation=1,
        mode="seg",
        pretrained=True,
        attn_pool=False,
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

        if pretrained is not None:
            self.load_state_dict(remove_fc(model_zoo.load_url(model_arch.url)))
        else:
            self.load_state_dict(torch.load(pretrained))

        self.attn_pool = None
        self.out_channels = 512 * block.expansion
        if attn_pool:
            self.attn_pool = nn.AdaptiveAvgPool2d((1, 1))
        #             output_dim = 512  # FIXME: need flexible
        #             input_dim = (384, 128)
        #             embed_dim = self.out_channels
        #             self.out_channels = output_dim
        #             down_ratio = 16 if res5_stride == 1 else 32
        #             num_heads = 32
        #             spacial_dim = (input_dim[0] // down_ratio, input_dim[1] // down_ratio)
        #             self.attn_pool = AttentionPool2d(
        #                 spacial_dim, embed_dim, num_heads, 2, output_dim
        #             )

        self.mode = mode
        if self.mode in ["seg", "segpool"]:
            self.out_channels = 256 * block.expansion
        if self.mode == "seg":
            del self.layer4

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

        if self.mode == "seg":
            return x
        y = self.layer4(x)
        if self.mode == "segpool":
            return y, x
        if self.attn_pool is not None:
            y = self.attn_pool(y).squeeze()
        return y

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
    mode = cfg.MODEL.EMBEDDING.EMBED_HEAD
    attn_pool = cfg.MODEL.RESNET.ATTN_POOL
    pretrained = cfg.MODEL.RESNET.PRETRAINED

    model_arch = model_archs[arch]
    model = ResNet(
        model_arch,
        res5_stride,
        res5_dilation,
        mode,
        pretrained=pretrained,
        attn_pool=attn_pool,
    )

    if cfg.MODEL.FREEZE:
        for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
