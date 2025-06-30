import os
import torch
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob

class SpectrumDataset(Dataset):
    def __init__(self, root_dir, params_file, transform=None, normalize=False, features=["E","perc_N","P","gain","ms"], exclude=[]):
        """
        Dataset for loading 1D spectrum data with experimental parameters.
        
        Args:
            root_dir (str): Directory containing the spectrum CSV files in subfolders
            params_file (str): Path to the params.csv file containing experimental parameters
            transform (callable, optional): Optional transform to be applied to the spectrum
            normalize (bool): Whether to normalize the intensity values to [0, 1]
        """
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize
        self.features = features
        self.params_df = pd.read_csv(params_file, engine='python')[features]# .apply(self.min_max_norm)
        self.exclude = exclude
        # Get all CSV files recursively
        self.file_list = self.get_list_of_files()
        self.file_list.sort()  # Ensure consistent ordering
        
        # Load first file to get spectrum length
        first_spectrum = pd.read_csv(self.file_list[0])
        self.spectrum_length = len(first_spectrum)
        
        # Calculate normalization values if needed
        if normalize:
            max_intensity = float('-inf')
            for file in self.file_list:
                spectrum = pd.read_csv(file)
                max_intensity = max(max_intensity, spectrum['intensity'].max())
            self.max_intensity = max_intensity
        
    def __len__(self):
        return len(self.file_list)

    def get_list_of_files(self, regex="*.csv"):
        files = []
        for dirpath, _, _ in os.walk(self.root_dir):
            if dirpath in self.exclude:
                print("Excluding " + dirpath)
                continue
            type_files = glob(os.path.join(dirpath, regex))
            files += type_files
        return sorted(files)

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #     filename = self.file_list[idx]
    #     image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    #     exp_index = int(filename.split('/')[-2])
    #     settings = self.settings.iloc[exp_index-1, 0:]
    #     settings = np.array([settings])
    #     settings = settings.astype('float32').reshape(-1, len(self.features))
    #     if self.transform:
    #         image = self.transform(image)
    #     to_tens = torchvision.transforms.ToTensor()
    #     settings = to_tens(settings).squeeze()
    #     sample = {'image': image, 'settings': settings}
    #     return sample
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load spectrum from CSV
        filename = self.file_list[idx]
        spectrum_df = pd.read_csv(filename)
        
        # Get experiment number from filename (assuming filename contains experiment number)
        exp_index = int(filename.split('/')[-2])
        
        # Get corresponding parameters
        settings = self.params_df.iloc[exp_index-1, 0:]
        settings = np.array([settings])
        settings = settings.astype('float32').reshape(-1, len(self.features))
        to_tens = torchvision.transforms.ToTensor()
        settings = to_tens(settings).squeeze()
        
        # Convert to tensor
        intensity = torch.tensor(spectrum_df['intensity'].values, dtype=torch.float32)
        energy = torch.tensor(spectrum_df['energy'].values, dtype=torch.float32)
        
        # Normalize intensity if requested
        if self.normalize:
            intensity = (intensity / self.max_intensity) * 2 - 1
            
        # Apply any additional transforms
        if self.transform:
            intensity = self.transform(intensity)
        # Use only the first 256 points from the spectrum
        return {
            'intensity': intensity[:256],
            'energy': energy[:256],
            'settings': settings
        }

def get_spectrum_dataloader(
    data_dir,
    params_file,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    transform=None,
    normalize=False,
    features=["E","perc_N","P","gain","ms"],
):
    """
    Creates a DataLoader for the spectrum dataset.
    
    Args:
        data_dir (str): Directory containing the spectrum CSV files
        params_file (str): Path to the params.csv file
        batch_size (int): Batch size for the dataloader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        transform (callable, optional): Optional transform to be applied to the spectra
        normalize (bool): Whether to normalize the intensity values
        
    Returns:
        DataLoader: PyTorch DataLoader for the spectrum dataset
    """
    dataset = SpectrumDataset(
        root_dir=data_dir,
        params_file=params_file,
        transform=transform,
        normalize=normalize,
        features=features
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

# Example usage:
if __name__ == "__main__":
    # Example of how to use the dataset
    data_dir = "../data/spectra"  # Directory containing the spectrum CSV files
    params_file = "../data/params.csv"
    
    # Create dataset
    dataset = SpectrumDataset(data_dir, params_file)
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample intensity shape: {sample['intensity'].shape}")
    print(f"Sample energy shape: {sample['energy'].shape}")
    
    # Create dataloader
    dataloader = get_spectrum_dataloader(data_dir, params_file, batch_size=32)
    
    # Example iteration
    for batch in dataloader:
        print(f"Batch intensity shape: {batch['intensity'].shape}")
        print(f"Batch energy shape: {batch['energy'].shape}")
        print(f"Batch settings shape: {batch['settings'].shape}")
        break
