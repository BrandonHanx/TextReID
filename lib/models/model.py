import clip
from torch import nn

from .backbones.bert import build_bert
from .backbones.deit import build_vit
from .backbones.gru import build_gru
from .backbones.hbgru import build_hbgru
from .backbones.hbvit import build_hybrid_vit
from .backbones.ibn_a import build_ibn_a
from .backbones.lstm import build_lstm
from .backbones.m_resnet import build_m_resnet
from .backbones.pit import build_pit
from .backbones.pvt import build_pvt
from .backbones.resnet import build_resnet
from .backbones.sagru import build_sagru
from .backbones.swin import build_swin
from .backbones.text_cnn import build_text_cnn
from .backbones.trans_encoder import build_trans_encoder
from .embeddings import build_embed
from .embeddings.moco_head.head import build_moco_head


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att_text_together = False
        self.tokenizer = None
        self.visual_model = None
        self.textual_model = None

        if cfg.MODEL.EMBEDDING.EMBED_HEAD == "clip":
            model, _ = clip.load("RN101", device="cuda", jit=False)
            model = model.float()
            self.visual_model = model.encode_image
            self.textual_model = model.encode_text
            self.embed_model = build_embed(
                cfg, model.visual.output_dim, model.visual.output_dim
            )
            self.tokenizer = clip.tokenize
        elif cfg.MODEL.EMBEDDING.EMBED_HEAD == "moco":
            if cfg.MODEL.VISUAL_MODEL in [
                "resnet50_ibn_a",
                "resnet101_ibn_a",
                "resnet34_ibn_a",
            ]:
                visual_model = build_ibn_a(cfg)
            elif cfg.MODEL.VISUAL_MODEL in [
                "resnet50",
                "resnet101",
            ]:
                visual_model = build_resnet(cfg)
            else:
                visual_model = build_m_resnet(cfg)
            textual_model = build_gru(cfg, bidirectional=True)
            self.embed_model = build_moco_head(cfg, visual_model, textual_model)
        else:
            if cfg.MODEL.VISUAL_MODEL in ["resnet34", "resnet50", "resnet101"]:
                self.visual_model = build_resnet(cfg)
            elif cfg.MODEL.VISUAL_MODEL == "clip_resnet":
                model, _ = clip.load("RN50", device="cuda", jit=False)
                model = model.float()
                self.visual_model = model.visual
                self.visual_model.out_channels = model.visual.output_dim
                del model.transformer
                del model.token_embedding
                del model.positional_embedding
                del model.ln_final
                del model.text_projection
                del model.logit_scale
            elif cfg.MODEL.VISUAL_MODEL in [
                "resnet50_ibn_a",
                "resnet101_ibn_a",
                "resnet34_ibn_a",
            ]:
                self.visual_model = build_ibn_a(cfg)
            elif cfg.MODEL.VISUAL_MODEL in ["m_resnet", "m_resnet50", "m_resnet101"]:
                self.visual_model = build_m_resnet(cfg)
            elif cfg.MODEL.VISUAL_MODEL == "hbvit":
                self.visual_model = build_hybrid_vit(cfg)
            elif cfg.MODEL.VISUAL_MODEL == "pit":
                self.visual_model = build_pit(cfg)
            elif cfg.MODEL.VISUAL_MODEL == "pvt":
                self.visual_model = build_pvt(cfg)
            elif cfg.MODEL.VISUAL_MODEL == "swin":
                self.visual_model = build_swin(cfg)
            else:
                self.visual_model = build_vit(cfg)
            if cfg.MODEL.TEXTUAL_MODEL == "bilstm":
                self.textual_model = build_lstm(cfg, bidirectional=True)
            elif cfg.MODEL.TEXTUAL_MODEL == "bigru":
                self.textual_model = build_gru(cfg, bidirectional=True)
            elif cfg.MODEL.TEXTUAL_MODEL == "transen":
                self.textual_model = build_trans_encoder(cfg)
            elif cfg.MODEL.TEXTUAL_MODEL == "hbgru":
                self.att_text_together = True
                self.textual_model = build_hbgru(cfg)
            elif cfg.MODEL.TEXTUAL_MODEL == "text_cnn":
                self.textual_model = build_text_cnn(cfg)
            elif cfg.MODEL.TEXTUAL_MODEL == "sagru":
                self.textual_model = build_sagru(cfg)
            else:
                self.textual_model = build_bert(cfg)
            self.embed_model = build_embed(
                cfg, self.visual_model.out_channels, self.textual_model.out_channels
            )
        self.use_att = cfg.DATASETS.USE_ATT
        self.whole = cfg.MODEL.WHOLE

    def forward(self, images, captions):
        # FIXME: more elegant
        #         self.textual_model.model.save_pretrained("./")
        if self.visual_model is None:
            return self.embed_model.forward(images, captions)
        visual_feat = self.visual_model(images)
        # if self.whole and not self.use_att:
        #     textual_feat, text_mask = self.textual_model(captions)
        #     captions = tuple([captions, text_mask])  # FIXME: need optimization
        if self.att_text_together and self.whole:
            textual_feat, mask, att_nums = self.textual_model(captions)
        elif self.tokenizer is not None:
            texts = [caption.text for caption in captions]
            texts = self.tokenizer(texts).cuda()
            textual_feat = self.textual_model(texts)
        else:
            textual_feat = self.textual_model(captions)

        if self.use_att and not self.att_text_together:
            if self.whole:
                attributes, att_nums = [], []
                for caption in captions:
                    attribute = caption.get_field("attribute")
                    attributes.append(attribute)
                    att_nums.append(len(attribute))
                attribute_feat = self.textual_model(attributes, equal_length=False)
                outputs_embed, losses_embed = self.embed_model(
                    visual_feat, textual_feat, attribute_feat, att_nums, captions
                )
            else:
                attributes = [caption.get_field("attribute") for caption in captions]
                attribute_feat = self.textual_model(attributes)

                outputs_embed, losses_embed = self.embed_model(
                    visual_feat, textual_feat, attribute_feat, captions
                )
        elif self.att_text_together and self.whole:
            outputs_embed, losses_embed = self.embed_model(
                visual_feat, textual_feat, mask, att_nums, captions
            )
        else:
            outputs_embed, losses_embed = self.embed_model(
                visual_feat, textual_feat, captions
            )

        if self.training:
            losses = {}
            losses.update(losses_embed)
            return losses

        return outputs_embed


def build_model(cfg):
    return Model(cfg)
