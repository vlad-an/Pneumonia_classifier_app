import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset,ConcatDataset
from sklearn.model_selection import train_test_split


def load_data(data_dir, batch_size=32, validation_split=0.2):
    """
    Load the dataset from specified train, val, and test directories.

    Parameters:
    - data_dir: Path to the data directory containing train, val, and test folders.
    - batch_size: The batch size for the DataLoader.

    Returns:
    - DataLoaders for the train, validation, and test sets.
    """
    
    # Define a general transformation that can be applied to all datasets.
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the image to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])
    
    # Load datasets from folders
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    
        # Combine the datasets
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    
    # Extract targets for stratification from the combined dataset
    # Note: ConcatDataset does not have a '.targets' attribute, so we need to combine targets manually
    combined_targets = np.concatenate([np.array(train_dataset.targets), np.array(val_dataset.targets)])
    
    # Split indices into train and validation indices using stratification
    train_indices, val_indices = train_test_split(
        range(len(combined_dataset)),
        test_size=validation_split,
        stratify=combined_targets,
        random_state=42  # for reproducibility
    )
    
    # Create DataLoader for train and validation sets using Subset
    train_loader = DataLoader(Subset(combined_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(combined_dataset, val_indices), batch_size=batch_size, shuffle=False)

    """
    # Concatenate all datasets
    all_dataset = train_dataset + val_dataset 
    
    # Shuffle the concatenated dataset
    np.random.seed(42)  # for reproducibility
    indices = np.random.permutation(len(all_dataset))
    
    # Split indices into train and validation indices
    split = int(np.floor(validation_split * len(all_dataset)))
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create DataLoader for train and validation sets using Subset
    train_loader = DataLoader(Subset(all_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(all_dataset, val_indices), batch_size=batch_size, shuffle=False)
    """
    
    return train_loader, val_loader
