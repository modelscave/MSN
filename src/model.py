import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MSNProcessor(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, mask_ratio=0.7):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Patchify is usually a Conv2d layer in ViT
        self.patch_embed = nn.Conv2d(
            in_channels=3, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def patchify(self, x):
        """Converts [B, 3, H, W] -> [B, Number_of_Patches, Embed_Dim]"""
        x = self.patch_embed(x)  # [B, C, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, L, C]
        return x

    def apply_mask(self, patches, mask_ratio=None):
        """Randomly drops patches based on the ratio."""
        ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        B, L, D = patches.shape
        keep_len = int(L * (1 - ratio))
        
        # Generate random indices to keep
        noise = torch.rand(B, L, device=patches.device)
        ids_keep = torch.argsort(noise, dim=1)[:, :keep_len]
        
        # Gather the kept patches
        # ids_keep is [B, keep_len], we need to expand it to match [B, keep_len, D]
        batch_indices = torch.arange(B, device=patches.device).unsqueeze(-1)
        masked_patches = patches[batch_indices, ids_keep]
        
        return masked_patches

    def forward(self, views_dict):
        """
        Processes the dictionary from your data loader:
        'target', 'main_anchor', 'focal_anchors'
        """
        # 1. Target View: Patchify ONLY (No Masking)
        target_patches = self.patchify(views_dict['target'])
        
        # 2. Main Anchor: Patchify + Masking
        main_anchor_patches = self.patchify(views_dict['main_anchor'])
        main_anchor_masked = self.apply_mask(main_anchor_patches)
        
        # 3. Focal Anchors: Patchify + Masking
        # Focal anchors are [B, M, 3, 96, 96]. We flatten B and M to process.
        B, M, C, H, W = views_dict['focal_anchors'].shape
        focals = views_dict['focal_anchors'].view(B * M, C, H, W)
        focal_patches = self.patchify(focals)
        focal_masked = self.apply_mask(focal_patches) 
        
        return {
            "target": target_patches,
            "main_anchor": main_anchor_masked,
            "focal_anchors": focal_masked.view(B, M, -1, focal_masked.shape[-1])
        }
    

class MSNEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', output_dim=256, use_prototypes=True, num_prototypes=1024):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        embed_dim = self.backbone.embed_dim
        
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )

        if use_prototypes:
            # We use a Linear layer but we will normalize its weights 
            # to treat it as a collection of prototype vectors.
            self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)
        else:
            self.prototypes = None

    def forward(self, x_patches, temperature=1.0):
        """
        x_patches: [B, L, D] 
        temperature: tau or tau+ from the paper
        """
        B, L, D = x_patches.shape

        # 1. Standard ViT Processing
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        
        pos_embed = self.backbone.pos_embed[:, 1:L+1, :]
        x = x_patches + pos_embed
        
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        
        # 2. Extract [CLS] and Project
        cls_rep = x[:, 0] 
        z = self.projector(cls_rep)
        
        # 3. L2 Normalize the representation (Crucial for Cosine Similarity)
        z = F.normalize(z, dim=-1, p=2)
        
        if self.prototypes is not None:
            # 4. L2 Normalize the Prototypes (Weights) - NO in-place modification
            # Get normalized prototypes without modifying the underlying parameter
            w_normalized = F.normalize(self.prototypes.weight, dim=1, p=2)
            
            # 5. Compute Logits: (z * prototypes^T)
            # Use the normalized weights directly
            logits = F.linear(z, w_normalized)
            
            # 6. Apply Temperature and Softmax as per Section 3
            # Formula: p = softmax(logits / temperature)
            p = F.softmax(logits / temperature, dim=-1)
            return p
        
        return z