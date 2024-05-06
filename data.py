import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.mask_paths, self.labels = self.load_paths_and_labels()


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]


        # Preprocess image and mask (resize with padding)
        image, mask = self.preprocess(image, mask)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, label

    def load_paths_and_labels(self):
        image_paths = []
        mask_paths = []
        labels = []
        for subdir in os.listdir(self.root_dir):
            if subdir in ['benign', 'normal']:
                image_dir = os.path.join(self.root_dir, subdir, 'images')
                mask_dir = os.path.join(self.root_dir, subdir, 'masks')
                for image_file in os.listdir(image_dir):
                    image_path = os.path.join(image_dir, image_file)
                    mask_file = image_file.replace('.png', '_mask.png')
                    mask_path = os.path.join(mask_dir, mask_file)
                    if os.path.exists(mask_path):
                        image_paths.append(image_path)
                        mask_paths.append(mask_path)
                        labels.append(subdir)  # Assigning label based on folder name
        return image_paths, mask_paths, labels

    def preprocess(self, image, mask):
        # Resize with padding to make dimensions divisible by 32
        height, width = image.shape[:2]
        new_width = int(np.ceil(width / 32) * 32)
        new_height = int(np.ceil(height / 32) * 32)
        pad_width = new_width - width
        pad_height = new_height - height
        pad_image = np.pad(image, ((pad_height // 2, pad_height - pad_height // 2),
                                   (pad_width // 2, pad_width - pad_width // 2)),
                           mode='constant', constant_values=0)
        pad_mask = np.pad(mask, ((pad_height // 2, pad_height - pad_height // 2),
                                 (pad_width // 2, pad_width - pad_width // 2)),
                          mode='constant', constant_values=0)
        return pad_image, pad_mask

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset
dataset = CustomDataset(root_dir='D:\\Development\\Python Projects\\Demo\\BCD_Project\\resized_images', transform=transform)

# Split dataset into train and validation sets
train_set, val_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Define data loaders
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
