import torch
import torch.nn as nn
import torch.nn.functional as F

class MSNLoss(nn.Module):
    def __init__(self, lambd=1.0):
        """
        lambd: weight of the ME-MAX regularization (default 1.0 per Appendix A.1)
        """
        super().__init__()
        self.lambd = lambd

    def forward(self, anchor_probs,focal_probs, target_probs):
        """
        anchor_probs: [B, K]      (Global masked views)
        focal_probs:  [B * M, K]  (Local focal views, e.g., 320, 1024)
        target_probs: [B, K]      (Original unmasked view)
        """
        # 1. Prepare Target Labels
        # Since target_probs are soft distributions, we use the cross-entropy 
        # formula: -sum(target * log(anchor))
        B = anchor_probs.shape[0]
        M = focal_probs.shape[0] // B  # 10

        # 1. Reshape focal_probs to [B, M, K] so we can join them with anchors
        # This groups the 10 focal views belonging to each of the 32 images
        focal_reshaped = focal_probs.view(B, M, -1) # [32, 10, 1024]

        # 2. Add a dimension to anchor_probs to make it [B, 1, K]
        anchor_reshaped = anchor_probs.unsqueeze(1) # [32, 1, 1024]

        # 3. Concatenate along the 'view' dimension (dim=1)
        # This creates [32, 11, 1024] where each row is [Anchor, F1, F2... F10]
        combined = torch.cat([anchor_reshaped, focal_reshaped], dim=1)

        # 4. Flatten back to [B * (M+1), K] -> [352, 1024]
        anchors = combined.view(-1, combined.shape[-1])

        # If anchor_probs contains focal views, we must repeat target_probs 
        # to match the number of anchor views per image (M+1)
        targets = target_probs.repeat_interleave(1+M, dim=0) # [B*(1+M), K]

        # 2. Cross-Entropy term: (1/MB) * sum(H(p+, p))
        # We use a small epsilon to avoid log(0)
        dot_product = torch.sum(targets * torch.log(anchors + 1e-8), dim=-1)
        ce_loss = -torch.mean(dot_product)

        # 3. ME-MAX Regularization: -lambda * H(p_bar)
        # Average prediction across the entire batch and all views
        p_bar = torch.mean(anchors, dim=0) # [K]
        
        # Entropy of the mean: sum(p_bar * log(p_bar))
        # Note: Eq (1) says -lambda * H(p_bar), and since H(p) = -sum(p log p), 
        # the term becomes +lambda * sum(p_bar * log(p_bar))
        me_max_loss = torch.sum(p_bar * torch.log(p_bar + 1e-8))
        
        # Total Loss = CE - lambda * Entropy
        total_loss = ce_loss + (self.lambd * me_max_loss)
        
        return total_loss