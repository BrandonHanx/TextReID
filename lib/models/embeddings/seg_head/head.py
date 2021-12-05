import torch
import torch.nn as nn

from lib.models.backbones.resnet import BasicBlock

from .loss import make_loss_evaluator


class SegHead(nn.Module):
    def __init__(self, cfg, visual_size, textual_size):
        super().__init__()
        self.visual_size = visual_size
        self.textual_size = textual_size
        self.res5_stride = cfg.MODEL.RESNET.RES5_STRIDE
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.num_parts = cfg.MODEL.NUM_PARTS

        # res neck
        self.res_global_branch = self._make_res_layer(
            self.visual_size, 2048, BasicBlock, 2, stride=self.res5_stride
        )
        self.res_local_branch_list = nn.ModuleList()
        for _ in range(self.num_parts):
            self.res_local_branch_list.append(
                self._make_res_layer(
                    self.visual_size, 256, BasicBlock, 2, stride=self.res5_stride
                )
            )
        # seg head
        self.seg_local_branch = self._make_seg_layer(inplanes=256, planes=256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.visual_embed_layer = nn.Linear(2048, self.embed_size)
        self.part_embed_layer = nn.Linear(256, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)
        self.attribute_embed_layer = nn.Linear(textual_size, self.embed_size)

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _make_res_layer(self, inplanes, planes, block, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_seg_layer(self, inplanes, planes):
        layers = []
        layers.append(nn.ConvTranspose2d(inplanes, planes, 2, 2, 0))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(planes, self.num_parts + 1, 1, 1, 0))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, attribute_feature, captions):
        batch_size = visual_feature.size(0)
        visual_feat = self.res_global_branch(visual_feature)

        part_feat_list = []
        seg_feat_list = []
        for i in range(self.num_parts):
            local = self.res_local_branch_list[i](
                visual_feature
            )  # b * 256 * h/16 * w/16
            seg_local = self.seg_local_branch(local)  # b * 6 * h/8 * w/8
            part_feat_list.append(local)
            seg_feat_list.append(seg_local)
        seg_feat = torch.cat(seg_feat_list, dim=0)  # 5b * 6 * h/8 * w/8

        visual_embed = self.avgpool(visual_feat)
        visual_embed = visual_embed.view(batch_size, -1)
        visual_embed = self.visual_embed_layer(visual_embed)
        part_embed_list = [
            self.maxpool(feature).view(batch_size, -1) for feature in part_feat_list
        ]
        part_embed_list = [
            self.part_embed_layer(feature) for feature in part_embed_list
        ]
        part_embed = torch.stack(part_embed_list, dim=0)

        textual_embed = textual_feature.view(batch_size, -1)
        textual_embed = self.textual_embed_layer(textual_embed)
        attribute_embed = attribute_feature.view(-1, self.textual_size)
        attribute_embed = self.attribute_embed_layer(attribute_embed)
        attribute_embed = attribute_embed.view(-1, batch_size, 256)

        if self.training:
            losses = self.loss_evaluator(
                visual_embed,
                textual_embed,
                part_embed,
                attribute_embed,
                seg_feat,
                captions,
            )
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        outputs.append(part_embed.permute(1, 0, 2))
        outputs.append(attribute_embed.permute(1, 0, 2))
        attributes = [caption.get_field("attribute") for caption in captions]
        tmask = torch.stack([attribute.get_field("mask") for attribute in attributes])
        outputs.append(tmask)
        return outputs, None


def build_seg_head(cfg, visual_size, textual_size):
    model = SegHead(cfg, visual_size, textual_size)
    return model
