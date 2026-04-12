import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from typing import cast

class TensorBoardAudioCallback(pl.Callback):
    """
    A modular callback to log audio to TensorBoard.
    Pulls a fixed sample from the Validation DataLoader to track true 
    generalization across epochs, making it safe for dynamic augmentation pipelines.
    """
    def __init__(self, log_every_n_epochs: int = 50, sample_rate: int = 16000):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.sample_rate = sample_rate
        self.fixed_noisy: torch.Tensor | None = None
        self.fixed_clean: torch.Tensor | None = None

    def on_fit_start(self, trainer, pl_module):
        """
        Called once at the very beginning of the trainer.fit() execution.
        We extract a single batch from the validation dataloader to serve as our anchor.
        """
        # Use getattr to bypass Pylance's missing type stubs for PyTorch Lightning
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("No datamodule on trainer")
        
        # 1. Grab the validation dataloader
        val_loader = datamodule.val_dataloader()
        
        # 2. Extract the first batch using an iterator
        noisy_batch, clean_batch = next(iter(val_loader))
        
        # 3. We only need one sample, not the whole batch. 
        # Extract the first tensor and add the batch dimension back.
        # Move it to the device the model is initializing on.
        self.fixed_noisy = noisy_batch[0].unsqueeze(0).to(pl_module.device) # Shape: (1, 1, 48000)
        self.fixed_clean = clean_batch[0].unsqueeze(0)

        # 4. Log the baseline ground-truth files to TensorBoard at step 0
        # Ensure logger exists and is specifically TensorBoard
        if trainer.logger is None:
            raise RuntimeError("No logger on trainer")
        tb_logger = cast(TensorBoardLogger, trainer.logger)
        tensorboard = tb_logger.experiment

        if self.fixed_noisy is None:
            raise RuntimeError("fixed_noisy tensor is None")
        
        tensorboard.add_audio(
            'Audio/1_Noisy_Input_Val', 
            self.fixed_noisy[0, 0], 
            global_step=0, 
            sample_rate=self.sample_rate
        )

        if self.fixed_clean is None:
            raise RuntimeError("fixed_clean tensor is None")
        
        tensorboard.add_audio(
            'Audio/2_Clean_Target_Val', 
            self.fixed_clean[0, 0], 
            global_step=0, 
            sample_rate=self.sample_rate
        )

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Hooking into the validation loop rather than the training loop 
        aligns the audio generation with the calculation of the val_loss.
        """
        # Only log every N epochs
        if trainer.current_epoch % self.log_every_n_epochs != 0 and trainer.current_epoch != 0:
            return
        
        if self.fixed_noisy is None:
            raise RuntimeError("fixed_noisy tensor is None")

        # Ensure the fixed tensor is on the correct device
        if self.fixed_noisy.device != pl_module.device:
            self.fixed_noisy = self.fixed_noisy.to(pl_module.device)

        # 1. The Forward Pass
        # We call the model directly to bypass any training-specific layers (like dropout)
        pl_module.eval() # Not strictly necessary, should already be in eval mode
        predicted_clean = pl_module(self.fixed_noisy)
        
        # 2. Log to TensorBoard
        # Ensure logger exists and is specifically TensorBoard
        if trainer.logger is None:
            raise RuntimeError("No logger on trainer")
        tb_logger = cast(TensorBoardLogger, trainer.logger)
        tensorboard = tb_logger.experiment

        tensorboard.add_audio(
            'Audio/3_Predicted_Output_Val', 
            predicted_clean[0, 0].cpu(), # Move back to CPU for logging
            global_step=trainer.global_step, 
            sample_rate=self.sample_rate
        )