import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from training.callbacks import TensorBoardAudioCallback
from model import EdgeAudioUSE
from datasets.voicebank_dataset import VoiceBankDataModule

# Run `tensorboard --logdir=tb_logs` to launch tensorboard server after script starts
# Server should be available at http://localhost:6006

# Run script from root as module with `python -m training.train`
def main():
    # Locks the random seed so that random cropping and weight initialization 
    # are 100% deterministic across multiple runs.
    pl.seed_everything(42)

    # Assumes we're running the script from the root python proj dir: `model/``
    CLEAN_DIR = "datasets/VoiceBank-DEMAND/clean_trainset_28spk_wav"
    NOISY_DIR = "datasets/VoiceBank-DEMAND/noisy_trainset_28spk_wav"

    if not os.path.exists(CLEAN_DIR) or not os.path.exists(NOISY_DIR):
        print("Error: Dataset directories not found. Please verify the paths.")
        return

    datamodule = VoiceBankDataModule(
        clean_dir=CLEAN_DIR, 
        noisy_dir=NOISY_DIR, 
        batch_size=8, 
        num_workers=4 # Adjust based on your MacBook's CPU core count
    )

    model = EdgeAudioUSE()

    # This will create a local folder structure: tb_logs/StageA_Overfit/version_X/
    logger = TensorBoardLogger("tb_logs", name="StageA_Overfit")
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss", # Monitoring training loss because we are overfitting
        mode="min",
        filename="use-overfit-{epoch:02d}-{train_loss:.4f}"
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            TensorBoardAudioCallback(log_every_n_epochs=50),
            checkpoint_callback
        ],
        max_epochs=500,
        
        # --- THE MAGIC OVERFIT FLAG ---
        # Lightning will ignore the rest of the dataset and train exclusively 
        # on the very first batch of 8 files. This isolates the U-Net math.
        overfit_batches=1,
        # By default, only writes the loss to TensorBoard every 50 batches to save hard drive space
        # When overfitting we only have one Epoch so we need to ask for logging every step
        log_every_n_steps=1,
        
        # Optional: If running on a MacBook with Apple Silicon (M1/M2/M3),
        # PyTorch Lightning can automatically use the MPS (Metal Performance Shaders) backend.
        accelerator="auto"
        
    )

    # Execute the training loop
    print("Starting Stage A: Mathematical Overfit Sanity Check...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()