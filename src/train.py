import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from math import cos, pi
import os
from datetime import datetime

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


