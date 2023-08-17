from .build import BACKBONE_REGISTRY
from lib.dinov2.vit import DinoVisionTransformer, vit_base, vit_large


@BACKBONE_REGISTRY.register()
def build_dino_v2_vit(cfg, input_shape):
    out_indices = cfg.DE.OUT_INDICES

    if out_indices is not None:
        if isinstance(out_indices, str):
            out_indices = [int(m) for m in out_indices.split(",")]
    
    if cfg.MODEL.BACKBONE.TYPE == 'small':
        return DinoVisionTransformer(
        patch_size=14,
        img_size=518,
        init_values=1,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        out_indices=out_indices,
    )
    elif cfg.MODEL.BACKBONE.TYPE == 'base':
        return vit_base(out_indices=out_indices)
    elif cfg.MODEL.BACKBONE.TYPE == "large":
        return vit_large(img_size=518, patch_size=14, init_values=1, out_indices=out_indices)
    else:
        raise NotImplementedError()