import torch
import torch.nn as nn
from terratorch.models.backbones.prithvi_vit import PrithviViT
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY


def build_model(num_classes=2, num_frames=2, freeze_encoder=False):
    """
    Build Prithvi-EO-2.0 model for binary classification.
    (Loads the Prithvi-EO-2.0 300M encoder (with temporal+location embeddings)

    Args:
        num_classes:    Number of output classes (2 for burned/destroyed and not-burned/not-destroyed)
        num_frames:     Number of input timestamps (2 for before/after pairs)
        freeze_encoder: If True, freeze the Prithvi encoder weights during training

    Returns:
        MyanmarClassifier model
    """
    encoder = TERRATORCH_BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300_tl",
        num_frames=num_frames,
        in_chans=6,
        img_size=224,
        pretrained=True,
    )


    """
    Freezing the encoder is useful when you have a small dataset.
    We don't want to overwrite Prithivi's carefully pretrained weights 
    with noise from only a few training examples -> we train the classification
    head instead, which has far fewer parameters.
    """
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    model = MyanmarClassifier(encoder, num_classes=num_classes)
    return model


class MyanmarClassifier(nn.Module):
    """
    Prithvi-EO-2.0 encoder + linear classification head.

    Input:  (B, num_frames, C, H, W) — e.g. (batch, 2, 6, 224, 224)
    Output: (B, num_classes)
    """

    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.embed_dim  # 1024 for 300M model
        # Classification head (single fully connected layer). It takes 1024-dimensional embedding and squashes it down to 2 numbers (one per class)
        self.head = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        # x: (B, num_frames, C, H, W)
        features = self.encoder(x)  # returns list of feature tensors
        # Take the last feature map and global average pool
        feat = features[-1]         # (B, num_patches, embed_dim)
        feat = feat.mean(dim=1)     # (B, embed_dim) — global average pool over patches
        return self.head(feat)      # (B, num_classes)


if __name__ == "__main__":
    model = build_model(num_classes=2, num_frames=2)
    print(model)

    # Test forward pass with a dummy before/after pair
    # 2 frames, 6 bands, 224x224
    dummy = torch.randn(2, 2, 6, 224, 224)  # batch=2
    out = model(dummy)
    print("Output shape:", out.shape)        # expect (2, 2)
