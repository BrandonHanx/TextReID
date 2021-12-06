from .gru import build_gru
from .m_resnet import build_m_resnet
from .resnet import build_resnet


def build_visual_model(cfg):
    if cfg.MODEL.VISUAL_MODEL in ["resnet50", "resnet101"]:
        return build_resnet(cfg)
    if cfg.MODEL.VISUAL_MODEL in ["m_resnet50", "m_resnet101"]:
        return build_m_resnet(cfg)
    raise NotImplementedError


def build_textual_model(cfg):
    if cfg.MODEL.TEXTUAL_MODEL == "bigru":
        return build_gru(cfg, bidirectional=True)
    raise NotImplementedError
