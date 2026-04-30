
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.optim as optim
from math import cos, pi
import argparse



from src.data import BinaryAMDDataset, RFMIDDataset, ODIRAMDDataset, MSNAugmentation, FundusDataset
from src.model import MSNEncoder, MSNProcessor
from src.train import train_msn
from src.loss import MSNLoss


def parse_arguments():
    parser = argparse.ArgumentParser(description='MSN Pretraining Script')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--total_iters', type=int, default=100000, help='Total training iterations')
    parser.add_argument('--warmup_iters', type=int, default=5000, help='Number of warmup iterations')
    parser.add_argument('--start_lr', type=float, default=0.0002, help='Starting learning rate for warmup')
    parser.add_argument('--base_lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-6, help='Final learning rate')
    parser.add_argument('--momentum_base', type=float, default=0.996, help='Base momentum for EMA')
    parser.add_argument('--weight_decay', type=float, default=0.04, help='Weight decay for optimizer')
    
    # Device and data
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for training (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--csv_file', type=str, default='/home/chaksuai/Yash/Data/ODIR-5k/ODIR-5K/ODIR-5K/data.xlsx',
                        help='Path to CSV file with data annotations')
    parser.add_argument('--paraquet_file', type=str, default='/home/chaksuai/Yash/Data/updated_fundus_descriptions_train.parquet',
                        help='Path to Parquet file with data annotations')
    parser.add_argument('--root_dir', type=str, default='/home/chaksuai/Yash/Data/ODIR-5k/ODIR-5K/ODIR-5K/Training Images',
                        help='Root directory containing training images')
    
    # Augmentation and model config
    parser.add_argument('--mask_ratio', type=float, default=0.7, help='Mask ratio for MSN augmentation')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension for patch embeddings')
    parser.add_argument('--num_focal_anchors', type=int, default=10, help='Number of focal anchors (m parameter)')
    parser.add_argument('--num_prototypes', type=int, default=1024, help='Number of prototypes')
    parser.add_argument('--output_dim', type=int, default=256, help='Output dimension for encoders')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='ViT model name')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for ViT')
    
    # Loss and checkpointing
    parser.add_argument('--lambd', type=float, default=1.0, help='Loss weight parameter')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save training logs')
    
    args = parser.parse_args()
    return args


def main(args):
    # --- 0. Hyperparameters and Config ---
    device = args.device

    # --- 1. Initialize Datasets and DataLoader ---
    msn_aug = MSNAugmentation(m=args.num_focal_anchors)
    # data = ODIRAMDDataset(csv_file=args.csv_file, root_dir=args.root_dir, transform=msn_aug)
    data = FundusDataset(parquet_file=args.paraquet_file, transform=msn_aug)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # --- 2. Initialize Models and Optimizer ---    
    anchor_encoder = MSNEncoder(model_name=args.model_name, output_dim=args.output_dim, use_prototypes=True, num_prototypes=args.num_prototypes)
    target_encoder = MSNEncoder(model_name=args.model_name, output_dim=args.output_dim, use_prototypes=True, num_prototypes=args.num_prototypes)
    processor = MSNProcessor(patch_size=args.patch_size, mask_ratio=args.mask_ratio, embed_dim=args.embed_dim)
    criterion = MSNLoss(lambd=args.lambd)

    optimizer = optim.AdamW(anchor_encoder.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    
    # --- 3. Train the MSN ---
    train_msn(
        anchor_encoder=anchor_encoder,
        target_encoder=target_encoder,
        dataloader=dataloader,
        processor=processor,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        total_iters=args.total_iters,
        warmup_iters=args.warmup_iters,
        start_lr=args.start_lr,
        base_lr=args.base_lr,
        final_lr=args.final_lr,
        momentum_base=args.momentum_base,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )   

if __name__ == "__main__":
    args = parse_arguments()
    main(args)