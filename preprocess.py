import os
import torch
import numpy as np
from PIL import Image
import h5py
import scipy.io as sio
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, data_dir, patch_size=128, transform=None, data_type='image'):
        """
        Dataset class for handling various data formats
        
        Args:
            data_dir (str): Directory containing the data files
            patch_size (int): Size of patches to extract
            transform (callable, optional): Transform to apply to the data
            data_type (str): Type of data files ('image', 'h5', 'matrix')
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.data_type = data_type
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get file list based on data type
        if data_type == 'image':
            self.file_list = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        elif data_type == 'h5':
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        elif data_type == 'matrix':
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith(('.npy', '.mat'))]
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def __len__(self):
        return len(self.file_list)

    def _load_image(self, file_path):
        """Load and process image file"""
        image = Image.open(file_path).convert('RGB')
        return self.transform(image)

    def _load_h5(self, file_path):
        """Load and process HDF5 file"""
        with h5py.File(file_path, 'r') as f:
            # Get the first dataset in the HDF5 file
            data = list(f.values())[0][:]
            # Convert to torch tensor and normalize
            if isinstance(data, np.ndarray):
                if data.ndim == 2:  # If 2D, add channel dimension
                    data = data[None, ...]
                elif data.ndim == 3 and data.shape[0] == 1:  # If 3D with single channel
                    pass
                elif data.ndim == 3 and data.shape[0] == 3:  # If 3D with RGB channels
                    pass
                else:
                    raise ValueError(f"Unsupported HDF5 data shape: {data.shape}")
                
                # Normalize to [0,1] range
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                # Convert to torch tensor
                data = torch.from_numpy(data).float()
                # Apply normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                data = (data - mean) / std
                return data
            else:
                raise ValueError("HDF5 data must be numpy array")

    def _load_matrix(self, file_path):
        """Load and process matrix file (.npy or .mat)"""
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        else:  # .mat file
            data = sio.loadmat(file_path)
            # Get the first array from the .mat file
            data = list(data.values())[-1]  # Usually the last value is the data array
        
        # Convert to torch tensor and normalize
        if isinstance(data, np.ndarray):
            if data.ndim == 2:  # If 2D, add channel dimension
                data = data[None, ...]
            elif data.ndim == 3 and data.shape[0] == 1:  # If 3D with single channel
                pass
            elif data.ndim == 3 and data.shape[0] == 3:  # If 3D with RGB channels
                pass
            else:
                raise ValueError(f"Unsupported matrix data shape: {data.shape}")
            
            # Normalize to [0,1] range
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            # Convert to torch tensor
            data = torch.from_numpy(data).float()
            # Apply normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            data = (data - mean) / std
            return data
        else:
            raise ValueError("Matrix data must be numpy array")

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        
        # Load data based on type
        if self.data_type == 'image':
            data = self._load_image(file_path)
        elif self.data_type == 'h5':
            data = self._load_h5(file_path)
        else:  # matrix
            data = self._load_matrix(file_path)
        
        # Extract random patch
        _, h, w = data.shape
        if h < self.patch_size or w < self.patch_size:
            data = transforms.Resize((self.patch_size, self.patch_size))(data)
        else:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            data = data[:, top:top + self.patch_size, left:left + self.patch_size]
        
        return data

def create_dataloader(data_dir, batch_size=1, patch_size=128, num_workers=4, data_type='image'):
    """
    Create a DataLoader for the dataset
    
    Args:
        data_dir (str): Directory containing the data files
        batch_size (int): Batch size for the dataloader
        patch_size (int): Size of patches to extract
        num_workers (int): Number of workers for data loading
        data_type (str): Type of data files ('image', 'h5', 'matrix')
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = ImageDataset(data_dir, patch_size=patch_size, data_type=data_type)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def preprocess_single_file(file_path, patch_size=128, data_type='image'):
    """
    Preprocess a single file for inference
    
    Args:
        file_path (str): Path to the input file
        patch_size (int): Size of patches to extract
        data_type (str): Type of data file ('image', 'h5', 'matrix')
    
    Returns:
        torch.Tensor: Preprocessed data tensor
    """
    if data_type == 'image':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data = Image.open(file_path).convert('RGB')
        data = transform(data)
    elif data_type == 'h5':
        with h5py.File(file_path, 'r') as f:
            data = list(f.values())[0][:]
            if isinstance(data, np.ndarray):
                if data.ndim == 2:
                    data = data[None, ...]
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                data = torch.from_numpy(data).float()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                data = (data - mean) / std
    else:  # matrix
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        else:  # .mat
            data = sio.loadmat(file_path)
            data = list(data.values())[-1]
        
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = data[None, ...]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            data = torch.from_numpy(data).float()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            data = (data - mean) / std
    
    # Pad if necessary
    _, h, w = data.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        data = transforms.Pad((0, 0, pad_w, pad_h))(data)
    
    return data.unsqueeze(0)  # Add batch dimension

def postprocess_data(tensor, data_type='image'):
    """
    Convert model output tensor back to original format
    
    Args:
        tensor (torch.Tensor): Model output tensor
        data_type (str): Type of output data ('image', 'h5', 'matrix')
    
    Returns:
        Union[PIL.Image, np.ndarray]: Processed data in original format
    """
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    
    # Clamp values to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    if data_type == 'image':
        return transforms.ToPILImage()(tensor.squeeze(0))
    else:  # h5 or matrix
        return tensor.squeeze(0).cpu().numpy() 