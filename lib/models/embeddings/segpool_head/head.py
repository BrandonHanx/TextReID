import torch
import torch.nn as nn

from lib.models.backbones.resnet import BasicBlock

from .loss import make_loss_evaluator


class SegPoolHead(nn.Module):
    def __init__(self, cfg, visual_size, textual_size):
        super().__init__()
        self.visual_size = visual_size
        self.textual_size = textual_size
        self.res5_stride = cfg.MODEL.RESNET.RES5_STRIDE
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.num_parts = cfg.MODEL.NUM_PARTS

        # local branch
        self.res_local_branch = self._make_res_layer(
            self.visual_size, 512, BasicBlock, 2, stride=self.res5_stride
        )
        # seg branch
        self.seg_branch = self._make_res_layer(
            self.visual_size, 256, BasicBlock, 2, stride=self.res5_stride
        )
        # seg head
        self.seg_head = self._make_seg_layer(inplanes=256, planes=256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_embed_layer = nn.Linear(2048, self.embed_size)
        self.part_embed_layer = nn.Linear(512, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)
        self.attribute_embed_layer = nn.Linear(textual_size, self.embed_size)

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    @staticmethod
    def _make_res_layer(inplanes, planes, block, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_seg_layer(self, inplanes, planes):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 1, 1, 0))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(planes, self.num_parts, 1, 1, 0))

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
        visual_feat, mid_feat = visual_feature  # use layer4 feature as global
        batch_size = mid_feat.size(0)
        visual_embed = self.avgpool(visual_feat).squeeze()
        visual_embed = self.visual_embed_layer(visual_embed)

        local_feat = self.res_local_branch(mid_feat)  # b * 512 * h/16 * w/16
        seg_feat = self.seg_branch(mid_feat)  # b * 256 * h/16 * w/16
        seg_feat = self.seg_head(seg_feat)  # b * 5 * h/16 * w/16

        if self.training:
            gt_masks = torch.stack([caption.get_field("crops") for caption in captions])

        part_embed_list = []
        for i in range(self.num_parts):
            if self.training:
                local_mask = gt_masks[:, i].unsqueeze(1)
            else:
                local_mask = torch.sigmoid(seg_feat[:, i]).unsqueeze(1)
            masked_local_feat = local_feat * local_mask
            part_embed_list.append(
                self.part_embed_layer(self.avgpool(masked_local_feat).squeeze())
            )
        part_embed = torch.stack(part_embed_list, dim=0)  # 5 * b * 256

        textual_embed = textual_feature.view(batch_size, -1)
        textual_embed = self.textual_embed_layer(textual_embed)
        attribute_embed = attribute_feature.view(-1, self.textual_size)
        attribute_embed = self.attribute_embed_layer(attribute_embed)
        attribute_embed = attribute_embed.view(-1, batch_size, 256)  # 5 * b * 256

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


def build_segpool_head(cfg, visual_size, textual_size):
    model = SegPoolHead(cfg, visual_size, textual_size)
    return model
