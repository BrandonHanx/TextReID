from .att_head.head import build_att_head
from .clip_head.head import build_clip_head
from .cross_head.head import build_cross_head
from .cutneg_head.head import build_cutneg_head
from .direct_hb_head.head import build_direct_hb_head
from .direct_head.head import build_direct_head
from .direct_i_head.head import build_direct_i_head
from .grl_head.head import build_grl_head
from .hungrian_head.head import build_hungrian_head
from .mha_head.head import build_mha_head
from .mmt_head.head import build_mmt_head
from .ot_head.head import build_ot_head
from .seg_head.head import build_seg_head
from .segpool_head.head import build_segpool_head
from .simple_head.head import build_simple_head
from .triplet_head.head import build_triplet_head


def build_embed(cfg, visual_out_channels, textual_out_channels):
    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "seg":
        return build_seg_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "segpool":
        return build_segpool_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "grl":
        return build_grl_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "mha":
        return build_mha_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "cross":
        return build_cross_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "direct":
        return build_direct_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "direct_hb":
        return build_direct_hb_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "direct_i":
        return build_direct_i_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "clip":
        return build_clip_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "hungrian":
        return build_hungrian_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "mmt":
        return build_mmt_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "ot":
        return build_ot_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "att":
        return build_att_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "cutneg":
        return build_cutneg_head(cfg, visual_out_channels, textual_out_channels)

    if cfg.MODEL.EMBEDDING.EMBED_HEAD == "triplet":
        return build_triplet_head(cfg, visual_out_channels, textual_out_channels)

    return build_simple_head(cfg, visual_out_channels, textual_out_channels)
