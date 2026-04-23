import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from math import cos, pi
import os
from datetime import datetime
from timm.loss import LabelSmoothingCrossEntropy

from .util import sinkhorn_normalization

def train_msn(
    anchor_encoder,    # Anchor f_theta
    target_encoder,    # Target f_theta_bar
    dataloader,
    processor, 
    optimizer, 
    criterion, 
    device,
    total_iters=100000,
    warmup_iters=5000,
    start_lr=0.0002,
    base_lr=1e-3,
    final_lr=1e-6,
    momentum_base=0.996,
    checkpoint_dir='./checkpoints',
    log_dir='./logs'
):
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Open log file and write header
    log_f = open(f"{log_dir}/training.log", 'w')
    log_f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"Total iterations: {total_iters}\n")
    log_f.write(f"Warmup iterations: {warmup_iters}\n")
    log_f.write("Step,Timestamp,LR,Loss\n")
    log_f.flush()
    
    # Ensure encoders are on device
    anchor_encoder.to(device)
    target_encoder.to(device)
    
    # Initialize Target weights with Anchor weights
    target_encoder.load_state_dict(anchor_encoder.state_dict())
    
    # Track best loss for model saving
    best_loss = float('inf')
    
    # Use an infinite generator for iteration-based training
    data_iter = iter(dataloader)
    
    for step in range(total_iters):
        # 1. Get Batch (and reset iterator if epoch ends)
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)

        # Update Scheduler (Cosine Decay for LR)
        curr_lr = adjust_learning_rate(optimizer, step, total_iters, warmup_iters, start_lr, base_lr, final_lr)
        
        # 2. Extract Views
        views = processor(images)  # Process the batch to get target, main_anchor, focal_anchors
        target_img = views['target'].to(device) # [B, 196, D]
        anchor_img = views['main_anchor'].to(device) # [B, 58, D] - Masked
        B, M, L_f, D = views['focal_anchors'].shape
        focal_imgs = views['focal_anchors'].view(B * M, L_f, D).to(device) # [B*M, 10, D] - Masked  
        
        # 3. Teacher Forward (EMA) - NO GRADIENTS
        with torch.no_grad():
            target_probs = target_encoder(target_img, temperature=0.025) # [B, K]
            target_probs = sinkhorn_normalization(target_probs, iterations=3, epsilon=0.05) # Apply Sinkhorn to get p+

        # 4. Student Forward
        anchor_probs = anchor_encoder(anchor_img, temperature=0.1) # [B, K]
        focal_probs = anchor_encoder(focal_imgs, temperature=0.1) # [B*M, K]
 
        # 5. Loss Calculation
        loss = criterion(anchor_probs, focal_probs, target_probs)

        # 6. Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7. Update Teacher via EMA Momentum Schedule
        with torch.no_grad():
            m = 1 - (1 - momentum_base) * (cos(pi * step / total_iters) + 1) / 2
            for param_q, param_k in zip(anchor_encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        if step % 100 == 0:
            log_msg = f"Step [{step}/{total_iters}] | LR: {curr_lr:.6f} | Loss: {loss.item():.4f}"
            print(log_msg)
            log_f.write(f"{step},{datetime.now().strftime('%H:%M:%S')},{curr_lr:.6f},{loss.item():.4f}\n")
            log_f.flush()
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'anchor_encoder': anchor_encoder.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'loss': loss.item()
            }, f"{checkpoint_dir}/best_model.pt")
        
        # Save periodic checkpoint
        if step % 10000 == 0 and step > 0:
            torch.save({
                'anchor_encoder': anchor_encoder.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
            }, f"{checkpoint_dir}/checkpoint_{step}.pt")
    
    # Save final model
    torch.save({
        'anchor_encoder': anchor_encoder.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': total_iters,
    }, f"{checkpoint_dir}/final_model.pt")
    
    # Close log file
    log_f.write(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.close()
    print(f"Training logs saved to {log_f.name}")

def adjust_learning_rate(optimizer, step, total_steps, warmup_steps, start_lr, base_lr, final_lr):
    """Cosine LR schedule with warmup."""
    
    if step < warmup_steps:
        lr = start_lr + (base_lr - start_lr) * step / warmup_steps
    else:
        step -= warmup_steps
        total_steps -= warmup_steps
        lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + cos(pi * step / total_steps))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


@torch.no_grad()
def validate_linear_probe(model, val_loader, processor, device):
    """Validate the linear probe model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    for images, targets in val_loader:
        images = processor(images).float()  # Apply processor and ensure float32
        images, targets = images.to(device), targets.to(device)
        
        logits = model(images)
        loss = criterion(logits, targets)
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, acc


def train_linear_probe(model, train_loader, val_loader, processor, total_iters=100000, 
                       learning_rate=0.1, momentum=0.9, val_freq=220, device='cuda',
                       checkpoint_dir='./checkpoints', log_dir='./logs'):
    """
    Train linear probe on top of frozen backbone with iteration-based training.
    Following MSN Appendix A.3 protocol.
    
    Args:
        model: LinearProbeModel
        train_loader: Training dataloader
        val_loader: Validation dataloader
        processor: IMGProcessor for patch extraction
        total_iters: Total number of training iterations
        learning_rate: SGD learning rate (default 0.1 per MSN Appendix A.3)
        momentum: SGD momentum (default 0.9)
        val_freq: Validation frequency in iterations
        device: Device to use
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
    """
    # Create checkpoint and log directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Open log file and write header
    log_f = open(f"{log_dir}/linear_probe_training.log", 'w')
    log_f.write(f"Linear Probe Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"Total iterations: {total_iters}\n")
    log_f.write(f"Validation frequency: {val_freq} iterations\n")
    log_f.write(f"Learning rate: {learning_rate}\n")
    log_f.write("Step,Timestamp,Train_Loss,Val_Loss,Val_Acc\n")
    log_f.flush()
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Move processor to device
    # processor = processor.to(device)
    
    # Only train the head
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    best_acc = 0
    best_loss = float('inf')
    data_iter = iter(train_loader)
    
    for step in range(total_iters):
        # 1. Get Batch (reset iterator if epoch ends)
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        
        images = processor(images).float()  # Apply the processor to all images and ensure float32
        images, targets = images.to(device), targets.to(device)
        
        # 2. Training step
        model.train()
        logits = model(images)
        loss = criterion(logits, targets)
        
        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss = loss.item()
        
        # 4. Validation step (iteration-based)
        if step % val_freq == 0 or step == total_iters - 1:
            model.eval()
            val_loss_total = 0
            val_correct = 0
            val_total = 0
            criterion_val = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for val_images, val_targets in val_loader:
                    val_images = processor(val_images).float()  # Apply processor
                    val_images, val_targets = val_images.to(device), val_targets.to(device)
                    
                    val_logits = model(val_images)
                    val_loss = criterion_val(val_logits, val_targets)
                    
                    val_loss_total += val_loss.item()
                    _, val_predicted = val_logits.max(1)
                    val_total += val_targets.size(0)
                    val_correct += val_predicted.eq(val_targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            val_loss_avg = val_loss_total / len(val_loader)
            
            log_msg = f"[Iter {step}/{total_iters}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%"
            print(log_msg)
            log_f.write(f"{step},{datetime.now().strftime('%H:%M:%S')},{train_loss:.4f},{val_loss_avg:.4f},{val_acc:.2f}\n")
            log_f.flush()
            
            # Save best model based on validation accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                best_loss = val_loss_avg
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step,
                    'val_acc': val_acc,
                    'val_loss': val_loss_avg
                }, f"{checkpoint_dir}/best_model.pt")
                print(f"  ✓ Best model saved with Val Acc: {val_acc:.2f}%")
            
            # Save periodic checkpoint
            if step % (val_freq * 10) == 0 and step > 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step,
                    'train_loss': train_loss,
                    'val_acc': val_acc,
                    'val_loss': val_loss_avg
                }, f"{checkpoint_dir}/checkpoint_{step}.pt")
        
        # Print training progress every 50 iterations
        elif step % 50 == 0 and step > 0:
            print(f"[Iter {step}/{total_iters}] Train Loss: {train_loss:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': total_iters,
        'best_val_acc': best_acc
    }, f"{checkpoint_dir}/final_model.pt")
    
    # Close log file
    log_f.write(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
    log_f.close()
    print(f"\nTraining logs saved to {log_dir}/linear_probe_training.log")
    print(f"Checkpoints saved to {checkpoint_dir}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    
    return model, best_acc
