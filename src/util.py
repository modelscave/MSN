import torch
from math import cos, pi
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .model import MSNEncoder



def load_pretrained_encoder(checkpoint_path, model_name, output_dim, num_prototypes, device):
    """Load pretrained target encoder from checkpoint"""
    print(f"Loading pretrained encoder from {checkpoint_path}...")
    
    encoder = MSNEncoder(model_name=model_name, output_dim=output_dim, 
                         use_prototypes=True, num_prototypes=num_prototypes)
    encoder.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load target_encoder weights
    if 'target_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['target_encoder'])
        print("Loaded target_encoder weights")
    elif 'anchor_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['anchor_encoder'])
        print("Loaded anchor_encoder weights")
    else:
        encoder.load_state_dict(checkpoint)
        print("Loaded encoder weights")
    
    return encoder


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def sinkhorn_normalization(logits, iterations=3, epsilon=0.05):
    """
    logits: [B, K] - The raw similarities from target_encoder (z * prototypes^T)
    iterations: Number of normalization steps (usually 3 is enough)
    epsilon: Entropic regularization parameter
    """
    # Q is the transpose of the similarity matrix
    # We work in log space for numerical stability
    Q = torch.exp(logits / epsilon).t() 
    B = Q.shape[1]  # Batch size
    K = Q.shape[0]  # Number of prototypes

    # Make the sum of all elements equal to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(iterations):
        # Normalize columns: sum of each column should be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

        # Normalize rows: sum of each row should be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

    # Normalize by B to return probabilities that sum to 1 per sample
    Q *= B 
    return Q.t() # [B, K]