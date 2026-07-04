import os
import random
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class VoiceBankDemandDataset(Dataset):
    """
    Ingests paired clean and noisy audio files.
    Enforces a strict 16kHz sample rate and yields fixed-length 3-second chunks 
    via random cropping to act as implicit data augmentation.
    """
    def __init__(self, clean_dir: str, noisy_dir: str, target_sr: int = 16000, chunk_seconds: float = 3.0):
        super().__init__()
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.target_sr = target_sr
        self.chunk_length_samples = int(target_sr * chunk_seconds)

        # VoiceBank clean and noisy files share the exact same filenames.
        # We read the filenames from the clean directory and assume they exist in the noisy directory.
        # Add sorted() to guarantee cross-platform reproducibility
        # self.filenames = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])

        # Overfit override to ensure we use the same wav file every run.
        self.filenames = ["p259_330.wav", "p259_330.wav"]
        
        if len(self.filenames) == 0:
            raise RuntimeError(f"No .wav files found in {clean_dir}")

        # Instantiate a reusable resampler.
        # Note: On-the-fly resampling in Python is computationally expensive. 
        # For Stage C (Production), we will pre-resample the dataset offline to save CPU cycles.
        # But for Stage A/B PoC, doing it dynamically is perfectly fine.
        self.resampler_48k_to_16k = torchaudio.transforms.Resample(orig_freq=48000, new_freq=self.target_sr)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # print(f"\n[DEBUG] PyTorch is fetching: {filename}")
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        # torchaudio loads raw waveforms as (Channels, Time)
        clean_wav, sr_clean = torchaudio.load(clean_path)
        noisy_wav, sr_noisy = torchaudio.load(noisy_path)

        # 1. Enforce 16kHz constraint (Wideband VoIP standard)
        if sr_clean != self.target_sr:
            clean_wav = self.resampler_48k_to_16k(clean_wav)
        if sr_noisy != self.target_sr:
            noisy_wav = self.resampler_48k_to_16k(noisy_wav)

        # 2. Extract a fixed 3-second chunk for batch uniformity
        current_length = clean_wav.shape[-1]

        if current_length > self.chunk_length_samples:
            # Random Cropping: Pick a random starting point.
            # This ensures the model rarely sees the exact same slice of audio twice across epochs.

            start_idx = 0 # @Overfit-Config: override to start at 0 for overfitting run, so model sees same chunk every time
            # start_idx = random.randint(0, current_length - self.chunk_length_samples)
            end_idx = start_idx + self.chunk_length_samples
            
            # Slice both tensors identically to maintain acoustic alignment
            clean_wav = clean_wav[..., start_idx:end_idx]
            noisy_wav = noisy_wav[..., start_idx:end_idx]
            
        elif current_length < self.chunk_length_samples:
            # Zero-Padding: If the file is shorter than 3 seconds, fill the rest with absolute silence.
            # This trains the model to output digital zero when no speech is present, preventing static hallucinations.
            pad_amount = self.chunk_length_samples - current_length
            clean_wav = F.pad(clean_wav, (0, pad_amount))
            noisy_wav = F.pad(noisy_wav, (0, pad_amount))

        # Output shape: (1, 48000), (1, 48000)
        return noisy_wav, clean_wav


class VoiceBankDataModule(pl.LightningDataModule):
    """
    Encapsulates the dataset logic for PyTorch Lightning.
    Handles the instantiation of the Dataset and DataLoaders cleanly so 
    our train.py script remains free of boilerplate.
    """
    def __init__(self, clean_dir: str, noisy_dir: str, batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # For our Stage A Overfit and Stage B PoC, we will use the exact same dataset 
        # for both training and validation. (When we move to Stage C, we will implement 
        # a proper 90/10 split or load a dedicated validation folder here).
        self.train_dataset = VoiceBankDemandDataset(
            clean_dir=self.clean_dir, 
            noisy_dir=self.noisy_dir
        )
        self.val_dataset = VoiceBankDemandDataset(
            clean_dir=self.clean_dir, 
            noisy_dir=self.noisy_dir
        )

    def train_dataloader(self):
        # shuffle=True ensures the batches are randomized every epoch
        # drop_last=True drops the final batch if it isn't perfectly sized, 
        # which prevents PyTorch from crashing on batch-size mismatches.
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        # Validation doesn't need to be shuffled.
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True
        )