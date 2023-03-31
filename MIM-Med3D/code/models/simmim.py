from typing import Union, Sequence
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

import numpy as np
# from .swin_3d import SwinTransformer3D
from .swin_unetr import SwinTransformer, PatchMerging, PatchMergingV2
from monai.networks.layers import Conv
from monai.networks.nets import ViT
from mmcv.runner import load_checkpoint
from timm.models.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")


class ViTSimMIM(nn.Module):
    def __init__(
        self,
        pretrained: Union[str, None],
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        masking_ratio: float = 0.5,
        revise_keys=[("model.", "")],
        **kwargs,
    ):
        super().__init__()

        self.pretrained = pretrained
        self.spatial_dims = spatial_dims

        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        self.encoder = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        # patch embedding block
        self.to_patch, self.patch_to_emb = self.encoder.patch_embedding.patch_embeddings
        n_patches = self.encoder.patch_embedding.n_patches
        patch_dim = self.encoder.patch_embedding.patch_dim

        # simple linear head
        self.mask_token = nn.Parameter(torch.randn(hidden_size))
        self.to_pixels = nn.Linear(hidden_size, patch_dim)

        self.init_weights(revise_keys=revise_keys)

    def init_weights(self, pretrained=None, revise_keys=[]):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)

            load_checkpoint(
                self,
                filename=self.pretrained,
                map_location=torch.device("cpu"),
                strict=False,
                revise_keys=revise_keys,
            )
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a path(str) or None")

    def forward(self, img):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.patch_embedding.position_embeddings

        # for indexing purposes
        batch_range = torch.arange(batch, device=device)[:, None]

        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_patches)
        mask_tokens = mask_tokens + self.encoder.patch_embedding.position_embeddings

        # calculate of patches needed to be masked, and get positions (indices) to be masked
        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = (
            torch.rand(batch, num_patches, device=device)
            .topk(k=num_masked, dim=-1)
            .indices
        )
        masked_bool_mask = (
            torch.zeros((batch, num_patches), device=device)
            .scatter_(-1, masked_indices, 1)
            .bool()
        )

        # mask tokens
        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        encoded = tokens

        # get the masked tokens
        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values
        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # # get the masked patches for the final reconstruction loss
        # masked_patches = patches[batch_range, masked_indices]

        # # calculate reconstruction loss
        # recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked

        return pred_pixel_values, patches, batch_range, masked_indices

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

class SwinSimMIM(nn.Module):
    def __init__(
        self,
        img_size: Sequence[int],
        in_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
        pretrained=None,
        revise_keys=[],
        masking_ratio=0.75,
        **kwargs,
    ):
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")
        
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        
        self.normalize = normalize
        self.masking_ratio = masking_ratio

        self.encoder = SwinTransformer(in_chans=in_channels,
                                        embed_dim=feature_size,
                                        window_size=window_size,
                                        patch_size=patch_size,
                                        depths=depths,
                                        num_heads=num_heads,
                                        mlp_ratio=4.0,
                                        qkv_bias=True,
                                        drop_rate=drop_rate,
                                        attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=dropout_path_rate,
                                        norm_layer=nn.LayerNorm,
                                        use_checkpoint=use_checkpoint,
                                        spatial_dims=spatial_dims,
                                        downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
                                        use_v2=use_v2,
                                        pretrained=pretrained,
                                        revise_keys=revise_keys)

        # patch embedding block
        
        self.num_features = self.encoder.embed_dim * 2 ** (self.encoder.num_layers)
        self.num_layers = self.encoder.num_layers
        self.final_resolution = (img_size[0] // 2 ** (self.num_layers+1),
                                 img_size[1] // 2 ** (self.num_layers+1),
                                 img_size[2] // 2 ** (self.num_layers+1))
        

        # masked tokens
        self.mask_token = nn.Parameter(torch.randn(feature_size))

        # simple linear head
        conv_trans = Conv[Conv.CONVTRANS, 3]
        self.conv3d_transpose = conv_trans(
            self.num_features,
            in_channels,
            kernel_size=(
                2 ** (self.num_layers + 1),
                2 ** (self.num_layers + 1),
                2 ** (self.num_layers + 1),
            ),
            stride=(2 ** (self.num_layers + 1), 2 ** (self.num_layers + 1), 2 ** (self.num_layers + 1),),
        )
        
    def forward(self, img):
        # B, C, D, H, W = img.shape
        device = img.device

        # get patches
        patches = rearrange(
            img,
            "b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)",
            p1=self.encoder.patch_size[0],
            p2=self.encoder.patch_size[1],
            p3=self.encoder.patch_size[2],
        ) # (B, num_patches, (p1 p2 p3 c))
        
        # swin_3d_forward
        tokens = self.encoder.patch_embed(img) # (B, embed_dim, x, x, x)
        
        # Start mask process.........
        tokens = tokens.permute(0, 2, 3, 4, 1) # (B, x, x, x, embed_dim)
        batch, x_1, x_2, x_3, embed_dim = tokens.shape
        tokens = tokens.view(batch, x_1 * x_2 * x_3, embed_dim)
        # (B, num_paches, embed_dim)
        batch, num_patches, *_ = tokens.shape
        assert num_patches == patches.shape[1]

        # for indexing purposes
        batch_range = torch.arange(batch, device=device)[:, None]

        # calculate of patches needed to be masked, and get positions (indices) to be masked
        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = (
            torch.rand(batch, num_patches, device=device)
            .topk(k=num_masked, dim=-1)
            .indices
        )
        masked_bool_mask = (
            torch.zeros((batch, num_patches), device=device)
            .scatter_(-1, masked_indices, 1)
            .bool()
        )

        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_patches)
        # mask_tokens = mask_tokens + self.encoder.patch_embedding.position_embeddings

        # mask tokens
        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens) # (B, num_patches, embeding_dim)
        tokens = tokens.view(batch, x_1, x_2, x_3, embed_dim) # (B, x, x, x, embed_dim)
        tokens = tokens.permute(0, 4, 1, 2, 3)
        
        # Finish mask , continue forward.......
        x0 = self.encoder.pos_drop(tokens) # (B, num_patches, embeding_dim)
        x0_out = self.encoder.proj_out(x0, self.normalize)
        if self.encoder.use_v2:
            x0 = self.encoder.layers1c[0](x0.contiguous())
        x1 = self.encoder.layers1[0](x0.contiguous())
        x1_out = self.encoder.proj_out(x1, self.normalize)
        if self.encoder.use_v2:
            x1 = self.encoder.layers2c[0](x1.contiguous())
        x2 = self.encoder.layers2[0](x1.contiguous())
        x2_out = self.encoder.proj_out(x2, self.normalize)
        if self.encoder.use_v2:
            x2 = self.encoder.layers3c[0](x2.contiguous())
        x3 = self.encoder.layers3[0](x2.contiguous())
        x3_out = self.encoder.proj_out(x3, self.normalize)
        if self.encoder.use_v2:
            x3 = self.encoder.layers4c[0](x3.contiguous())
        x4 = self.encoder.layers4[0](x3.contiguous())
        x4_out = self.encoder.proj_out(x4, self.normalize)
        tokens = x4_out
        
        # upsample
        pred_pixel_values = self.conv3d_transpose(tokens)

        pred_pixel_values = rearrange(
            pred_pixel_values,
            "b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)",
            p1=self.encoder.patch_size[0],
            p2=self.encoder.patch_size[1],
            p3=self.encoder.patch_size[2],
        )
        
        pred_pixel_values = pred_pixel_values[batch_range, masked_indices]
        

        return pred_pixel_values, patches, batch_range, masked_indices


if __name__ == "__main__":
    # config
    img_size = [96, 96, 96]
    in_channels = 1
    depths = [2, 2, 2, 2]
    num_heads = [3, 6, 12, 24]
    feature_size = 48
    drop_rate = 0.0
    attn_drop_rate = 0.0
    dropout_path_rate = 0.0 
    normalize = True
    use_checkpoint = False
    spatial_dims = 3
    downsample = "merging"
    use_v2 = False
    pretrained = None
    revise_keys = []
    
    
    model = SwinSimMIM(img_size,
                        in_channels,
                        depths,
                        num_heads,
                        feature_size,
                        drop_rate,
                        attn_drop_rate,
                        dropout_path_rate,
                        normalize,
                        use_checkpoint,
                        spatial_dims,
                        downsample,
                        use_v2,
                        pretrained,
                        revise_keys,
                        masking_ratio=0.75)

    x = torch.randn(2, 1, 96, 96, 96)
    y = model(x)

    print(y[0].shape)
