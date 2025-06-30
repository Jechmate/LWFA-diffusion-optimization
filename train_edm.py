import copy
import logging
import os
import numpy as np
from tqdm import tqdm

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.spectrum_dataset import *

from src.modules_1d import EMA, EDMPrecond
from src.diffusion import *
from src.utils import *
from src.dataset import *
from src.loss import EDMLoss

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def calculate_sigma_data(dataloader):
    all_intensities = []
    for batch in dataloader:
        all_intensities.append(batch['intensity'])
    
    data = torch.cat(all_intensities, dim=0)
    return torch.std(data).item()

def train(args, model=None, finetune=False):
    setup_logging(args.run_name)
    device = args.device

    data_dir = "data/spectra"  # Directory containing the spectrum CSV files
    params_file = "data/params.csv"
    
    # Create dataloader with exclusions
    exclude_paths = []
    if hasattr(args, 'current_exclude') and args.current_exclude is not None:
        exclude_paths = [os.path.join(data_dir, str(args.current_exclude))]
        print(f"Excluding experiment: {args.current_exclude}")
    
    # Create dataset with exclusions
    dataset = SpectrumDataset(data_dir, params_file, features=args.features, normalize=True, exclude=exclude_paths)
    
    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    sigma_data = calculate_sigma_data(train_dataloader)
    print(f"Sigma data: {sigma_data}")
    print(f"Training dataset size: {len(dataset)}")
    gradient_acc = args.grad_acc
    l = len(train_dataloader)
    steps_per_epoch = l / gradient_acc

    #---------------------------------------------------------------------------
    if not model:
        print("Training from scratch")
        model = EDMPrecond(
            resolution=args.length,
            settings_dim=len(args.features),
            sigma_min=0,
            sigma_max=float('inf'),
            sigma_data=sigma_data,
            model_type='UNet_conditional',
            device=device
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    #---------------------------------------------------------------------------

    sampler = EdmSampler(net=model, num_steps=10)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    loss_fn = EDMLoss()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    model.train().requires_grad_(True)
    
    # Keep track of epoch losses for running mean
    epoch_losses = []
    
    # Training loop with single progress bar over epochs
    pbar = tqdm(range(1, args.epochs + 1), desc=f"Training (excluding exp {args.current_exclude if hasattr(args, 'current_exclude') else 'none'})")
    
    for epoch in pbar:
        logging.info(f"Starting epoch {epoch}:")
        
        epoch_loss_sum = 0.0
        num_batches = 0

        for i, data in enumerate(train_dataloader):
            intensities = data['intensity'].to(device)
            settings = data['settings'].to(device)

            # Randomly drop conditioning for classifier-free guidance
            if np.random.random() < 0.1:
                settings = None
                
            # The intensity data needs to be reshaped to (batch, channels, length)
            intensities = intensities.unsqueeze(1)  # Add channel dimension
            
            loss = loss_fn(net=model, y=intensities, settings=settings)

            # Accumulate gradients
            optimizer.zero_grad()
            loss.mean().backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            # Update EMA
            ema.step_ema(ema_model, model)

            # Accumulate loss for epoch average
            epoch_loss_sum += loss.mean().item()
            num_batches += 1

            logger.add_scalar("Loss", loss.mean(), global_step=epoch * l + i)

        # Calculate epoch average loss
        epoch_avg_loss = epoch_loss_sum / num_batches
        epoch_losses.append(epoch_avg_loss)
        
        # Calculate running mean of last 10 epochs
        if len(epoch_losses) >= 10:
            running_mean_loss = np.mean(epoch_losses[-10:])
        else:
            running_mean_loss = np.mean(epoch_losses)
        
        # Update progress bar with running mean loss
        pbar.set_postfix({
            "Epoch Loss": f"{epoch_avg_loss:.4f}",
            "Running Mean (10)": f"{running_mean_loss:.4f}"
        })

        # Save samples periodically
        if args.sample_freq and epoch % args.sample_freq == 0:
            model.eval()
            with torch.no_grad():
                # Sample some spectra
                settings_sample = torch.tensor([args.sample_settings], dtype=torch.float32).to(device)
                ema_sampled_vectors = sampler.sample(
                    resolution=args.length,
                    device=device,
                    settings=settings_sample,
                    n_samples=args.n_samples,
                    cfg_scale=3,
                    settings_dim=len(args.features)
                )
                
                # Save the sampled spectrum
                np.save(os.path.join("results", args.run_name, f"sample_epoch_{epoch}.npy"), 
                       ema_sampled_vectors.cpu().numpy())
                
            model.train()
            
            # Save checkpoint
            torch.save(ema_model.state_dict(), os.path.join("models",
                                                            args.run_name,
                                                            f"ema_ckpt_epoch_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models",
                                                            args.run_name,
                                                            f"optim_epoch_{epoch}.pt"))

    # Save final checkpoints
    torch.save(ema_model.state_dict(), os.path.join("models",
                                                    args.run_name,
                                                    f"ema_ckpt_final.pt"))
    torch.save(optimizer.state_dict(), os.path.join("models",
                                                    args.run_name,
                                                    f"optim_final.pt"))
    
    # Generate final samples
    model.eval()
    with torch.no_grad():
        settings_sample = torch.tensor([args.sample_settings], dtype=torch.float32).to(device)
        ema_sampled_vectors = sampler.sample(
            resolution=args.length,
            device=device,
            settings=settings_sample,
            n_samples=args.n_samples,
            cfg_scale=3,
            settings_dim=len(args.features)
        )
        
        # Save the final sampled spectrum
        np.save(os.path.join("results", args.run_name, f"final_sample.npy"), 
               ema_sampled_vectors.cpu().numpy())

def train_with_exclusions(exclude_experiments=None):
    """
    Train models excluding specific experiments one at a time.
    
    Args:
        exclude_experiments (list): List of experiment numbers/folder names to exclude
    """
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Base configuration
    base_run_name = "edm_4kepochs"
    args.epochs = 4000
    args.n_samples = 4
    args.batch_size = 128
    args.length = 256
    args.features = ["E", "P", "ms"]
    args.device = "cuda:1"
    args.lr = 1e-3
    args.grad_acc = 1
    args.sample_freq = 1000
    args.data_path = "data/spectra"
    args.csv_path = "data/params.csv"
    args.sample_settings = [20, 15, 30]

    if exclude_experiments is None:
        # If no exclusions specified, train normally without exclusions
        args.run_name = base_run_name
        args.current_exclude = None
        print("Training without exclusions")
        train(args, model=None)
    else:
        # Train separate models for each exclusion
        for experiment in exclude_experiments:
            args.current_exclude = experiment
            args.run_name = f"{base_run_name}_exclude_{experiment}"
            
            print(f"\n{'='*50}")
            print(f"Training model excluding experiment: {experiment}")
            print(f"Model will be saved with run_name: {args.run_name}")
            print(f"{'='*50}\n")
            
            train(args, model=None)
            
            print(f"\nCompleted training for exclusion of experiment {experiment}")
            print(f"Model saved in: models/{args.run_name}/")

def launch():
    """
    Launch training with or without exclusions.
    Modify the exclude_experiments list to specify which experiments to exclude.
    """
    # Specify which experiments to exclude during training
    # Each number corresponds to a folder name in data/spectra/
    # Set to None or empty list to train without exclusions
    exclude_experiments = [3, 8, 11, 19, 21]  # Example: exclude experiments 3, 8, 11, 19, 21
    
    # Uncomment the line below to train without any exclusions
    # exclude_experiments = None
    
    train_with_exclusions(exclude_experiments)

if __name__ == '__main__':
    launch()