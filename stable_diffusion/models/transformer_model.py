# models/transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80,
                 embed_dim=768, num_layers=6, num_heads=8, norm_type="batch"):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Normalization layer choice (applied after projection, before classifier)
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(embed_dim)
        elif norm_type == "group":
            # GroupNorm requires num_channels divisible by num_groups
            self.norm = nn.GroupNorm(num_groups=8, num_channels=embed_dim)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(embed_dim)
        else:
            raise ValueError("Unsupported normalization type.")
            
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.encoder(x)
        
        # Global average pooling over patches
        x = x.mean(dim=1)
        
        # Apply chosen normalization (if using BatchNorm1d, ensure proper shape)
        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x)
        else:
            x = self.norm(x)
            
        logits = self.classifier(x)
        return logits

if __name__ == "__main__":
    # Quick test
    model = TransformerClassifier(norm_type="layer")
    dummy = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    print("Output shape:", output.shape)
