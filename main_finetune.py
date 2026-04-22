
import torch
import torchvision.transforms as T
from torch import nn
import argparse



from src.data import ODIRAMDDataset, RFMIDDataset
from src.model import MSNEncoder, LinearProbeModel, IMGProcessor
from src.train import train_linear_probe



def parse_arguments():
    parser = argparse.ArgumentParser(description='Linear Probe Training for Eye Disease Classification')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use for training (default: cuda)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='Name of the vision transformer model (default: vit_base_patch16_224)')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='Output dimension of the encoder (default: 256)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classification classes (default: 2)')
    parser.add_argument('--feature_dim', type=int, default=768,
                        help='Feature dimension for the model (default: 768)')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size for image processing (default: 16)')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Embedding dimension (default: 768)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/home/chaksuai/Yash/Masked Siemese Network/checkpoints/best_model.pt',
                        help='Path to the pre-trained checkpoint')
    parser.add_argument('--output_path', type=str,
                        default='/home/chaksuai/Yash/best_linear_probe.pt',
                        help='Path to save the trained model')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str,
                        default='/home/chaksuai/Yash/Data/ODIR-5k/ODIR-5K/ODIR-5K/data.xlsx',
                        help='Path to training dataset CSV file')
    parser.add_argument('--train_dir', type=str,
                        default='/home/chaksuai/Yash/Data/ODIR-5k/ODIR-5K/ODIR-5K/Training Images',
                        help='Path to training images directory')
    parser.add_argument('--val_csv', type=str,
                        default='/home/chaksuai/Yash/Data/A. RFMiD_All_Classes_Dataset/2. Groundtruths/c. RFMiD_Testing_Labels.csv',
                        help='Path to validation dataset CSV file')
    parser.add_argument('--val_dir', type=str,
                        default='/home/chaksuai/Yash/Data/A. RFMiD_All_Classes_Dataset/1. Original Images/c. Testing Set',
                        help='Path to validation images directory')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for training (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for optimizer (default: 0.9)')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Total number of iterations for training (default: 100000)')
    parser.add_argument('--val_freq', type=int, default=220,
                        help='Validation frequency in iterations (default: 220)')
    
    # Image parameters
    parser.add_argument('--image_size', type=int, default=224,
                        help='Target image size (default: 224)')
    
    # Logging and checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs (default: ./logs)')
    
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

device = args.device if torch.cuda.is_available() or 'cpu' in args.device else "cpu"
print(f"Using device: {device}")

# Step 1: Load pre-trained encoder
pretrained_encoder = MSNEncoder(
    model_name=args.model_name, 
    output_dim=args.output_dim, 
    use_prototypes=False
)

# Load checkpoint
checkpoint = torch.load(args.checkpoint_path, map_location=device)
pretrained_encoder.load_state_dict(checkpoint['target_encoder'], strict=False)
print(f"Loaded pre-trained encoder from {args.checkpoint_path}")

# Step 2: Create linear probe model
linear_probe_model = LinearProbeModel(
    backbone=pretrained_encoder,
    num_classes=args.num_classes,
    feature_dim=args.feature_dim
)
linear_probe_model = linear_probe_model.to(device)
print("Created linear probe model")

# Step 3: Prepare data loaders with standard transforms (no augmentation for linear probe)
transform_lp = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data_lp = ODIRAMDDataset(
    csv_file=args.train_csv,
    root_dir=args.train_dir,
    transform=transform_lp
)

val_data_lp = RFMIDDataset(
    csv_file=args.val_csv, 
    root_dir=args.val_dir, 
    transform=transform_lp
)

train_loader_lp = torch.utils.data.DataLoader(train_data_lp, batch_size=args.batch_size, shuffle=True)
val_loader_lp = torch.utils.data.DataLoader(val_data_lp, batch_size=args.batch_size, shuffle=False)

# Initialize processor
processor_lp = IMGProcessor(patch_size=args.patch_size, embed_dim=args.embed_dim)

print(f"Train samples: {len(train_data_lp)}, Val samples: {len(val_data_lp)}")

# Step 4: Run linear probing training
print("\n" + "="*70)
print("Starting Linear Probing Training...")
print("="*70 + "\n")

trained_model, best_acc = train_linear_probe(
    model=linear_probe_model,
    train_loader=train_loader_lp,
    val_loader=val_loader_lp,
    processor=processor_lp,
    total_iters=args.total_iters,
    learning_rate=args.learning_rate,
    momentum=args.momentum,
    val_freq=args.val_freq,
    device=device,
    checkpoint_dir=args.checkpoint_dir,
    log_dir=args.log_dir
)
