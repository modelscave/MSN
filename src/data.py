import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class BinaryAMDDataset(datasets.ImageFolder):
    def find_classes(self, directory):
        # Manually define the mapping regardless of folder order
        classes = ['non-amd', 'amd']
        class_to_idx = {
            'amd': 1,
            'cataract': 0,
            'diabetes': 0,
            'normal': 0
        }
        return classes, class_to_idx
    
class RFMIDDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data_frame.iloc[idx, 0]) + '.png')
        image = Image.open(img_name).convert('RGB') 
        label = self.data_frame['ARMD'][idx] 

        if self.transform:
            image = self.transform(image)

        return image, label

class ODIRAMDDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_excel(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Create AMD labels
        amd_keywords = ['dry age-related macular degeneration', 'wet age-related macular degeneration']
        self.data_frame['left_amd_label'] = self.data_frame['Left-Diagnostic Keywords'].apply(
            lambda x: 1 if any(keyword in str(x).lower() for keyword in amd_keywords) else 0
        )
        self.data_frame['right_amd_label'] = self.data_frame['Right-Diagnostic Keywords'].apply(
            lambda x: 1 if any(keyword in str(x).lower() for keyword in amd_keywords) else 0
        )

    def __len__(self):
        return 2 * len(self.data_frame)  # Two eyes per patient

    def __getitem__(self, idx):
        row = idx // 2
        is_left = idx % 2 == 0
        
        if is_left:
            img_name = os.path.join(self.root_dir, self.data_frame.iloc[row, 3])  # Left-Fundus column
            label = self.data_frame.iloc[row, -2]  # left_amd_label
        else:
            img_name = os.path.join(self.root_dir, self.data_frame.iloc[row, 4])  # Right-Fundus column
            label = self.data_frame.iloc[row, -1]  # right_amd_label
        
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label    

class FundusDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        self.dataframe = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['file_path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['modality']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MSNAugmentation:
    def __init__(self, m=10):
        self.m = m
        
        # 1. Standard SimCLR-style augmentations (Appendix A.1)
        # Includes: Random Resized Crop, Flip, Color Jitter, and Gaussian Blur
        self.base_aug = T.Compose([
            T.RandomResizedCrop(224, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. Focal views (96x96) - typically smaller scale crops
        self.focal_aug = T.Compose([
            T.RandomResizedCrop(96, scale=(0.05, 0.2)), # Small area for focal context
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        # Create Target View (224x224)
        target_view = self.base_aug(image)
        
        # Create Main Anchor View (224x224)
        # Note: Masking happens inside the model/forward pass, not here.
        main_anchor_view = self.base_aug(image)
        
        # Create 'm' Focal Anchor Views (96x96)
        focal_views = [self.focal_aug(image) for _ in range(self.m)]
        
        # Return as a dictionary or a list
        return {
            "target": target_view,
            "main_anchor": main_anchor_view,
            "focal_anchors": torch.stack(focal_views)
        }
