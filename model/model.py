import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import auraloss
from modules.unified_mamba_block import UnifiedMambaBlock
from layers import EncoderBlock, DecoderBlock

# --- The PyTorch Lightning Wrapper ---

class EdgeAudioUSE(pl.LightningModule):
    """
    Universal Speech Enhancer (USE) optimized for real-time edge inference.

    This model fuses a causal 1D Convolutional U-Net topology with a State Space Model 
    (Mamba) bottleneck. It is designed to perform simultaneous noise suppression, 
    dereverberation, and Bandwidth Extension (BWE) directly on raw waveforms, while 
    maintaining a sub-20ms algorithmic latency suitable for live VoIP communications.

    The architecture compresses raw audio into a dense latent space so the Mamba 
    blocks can track global acoustic context without thrashing the CPU's L1 cache, 
    before upsampling back to a clean waveform.

    Hyperparameters:
        channels_H (int): The base channel capacity of the network. The encoder will 
            progressively double this capacity at each layer (capped at 768) to 
            capture complex acoustic textures. Defaults to 64.
        encoder_layers (int): The depth of the U-Net. Dictates the temporal downsampling 
            ratio. 6 layers at a stride of 2 yields a 64x compression rate, ensuring 
            the 12ms target latency. Defaults to 6.
        d_model (int): The internal dimension of the Mamba bottleneck layers. 
            A larger dimension increases expressive power for complex reverb removal 
            but quadratically increases MACs. Defaults to 512.
        lookahead (int or str): The global causality toggle for the entire network. 
            - Strictly Causal (lookahead=0): Pads only the past. Guarantees zero algorithmic 
            latency for real-time edge inference.
            - Micro-Lookahead (lookahead>0): Sacrifices a specific number of frames (e.g., 2ms) 
            of latency to give the model future context. Used to test the BWE Causal Paradox 
            (whether the model needs to see the end of a transient to reconstruct high frequencies).
            - Non-Causal (lookahead='centered'): Traditional offline processing for theoretical baselines.
            Defaults to 0.
    """
    def __init__(self, channels_H=64, encoder_layers=6, d_model=512, lookahead=0):
        super().__init__()
        self.save_hyperparameters()
        self.lookahead = lookahead

        """
        Multi-Resolution STFT Loss (Frequency-Domain Perception)
        Calculates spectrogram loss across three distinct physical time windows (assuming 16kHz audio):
        1. win_length=240 (15ms): Micro-window. Forces accurate reconstruction of sharp transients/consonants.
        2. win_length=600 (37.5ms): Standard window. Captures general speech articulation.
        3. win_length=1200 (75ms): Macro-window. Forces accurate, phase-stable reconstruction of sustained vowels.
        The hop_sizes dictate an ~80% overlap to ensure smooth gradient flow during backpropagation.
        """
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[50, 120, 240],
            win_lengths=[240, 600, 1200]
        )
        
        # 1. The Encoder
        self.encoder = nn.ModuleList()
        in_c = 1
        out_c = channels_H
        for i in range(encoder_layers):
            self.encoder.append(EncoderBlock(in_c, out_c, lookahead=self.lookahead))
            in_c = out_c
            out_c = min(out_c * 2, 768) 

        # 2. The Mamba Bottleneck
        self.proj_in = nn.Conv1d(in_c, d_model, kernel_size=1)
        self.mamba_blocks = nn.ModuleList([
            UnifiedMambaBlock(d_model=d_model, d_state=64) for _ in range(3)
        ])
        self.proj_out = nn.Conv1d(d_model, in_c, kernel_size=1)

        # 3. The Decoder
        self.decoder = nn.ModuleList()
        for i in range(encoder_layers):
            # The '//' operator is Python's "floor division" (integer division). 
            # E.g., 128 // 2 = 64. 
            # The inline 'if/else' statement means: halve the channels for the next layer, 
            # UNLESS this is the very last layer, in which case force the output to 1 channel (mono audio).
            out_c = in_c // 2 if i < encoder_layers - 1 else 1
            # A boolean flag to tell the DecoderBlock whether or not to apply a ReLU activation.
            # You don't want a ReLU on the final audio output, otherwise it cuts off the negative half of the audio wave.
            is_last = (i == encoder_layers - 1)
            self.decoder.insert(0, DecoderBlock(in_c, out_c, is_last=is_last, lookahead=self.lookahead))
            in_c = out_c

    def forward(self, x):
        skips = []
        
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            
        skips = skips[::-1] 

        x = self.proj_in(x)
        # Mamba blocks need (Batch, Time, Channels)
        x = x.permute(0, 2, 1) 
        
        for mamba in self.mamba_blocks:
            x = mamba(x)
        
        # Swapping back to (Batch, Channels, Time)
        x = x.permute(0, 2, 1) 
        x = self.proj_out(x)

        for i, block in enumerate(self.decoder):
            skip = skips[i][..., :x.shape[-1]]
            x = x + skip
            x = block(x)

        return x

    def training_step(self, batch, batch_idx):
        noisy_audio, clean_audio = batch 
        denoised_audio = self(noisy_audio)

        """ Loss Derivation
        L1 loss alone matches the waveform mathematically but sounds "phasy" and unnatural. 
        Multi-Resolution STFT loss analyzes the audio in the frequency domain across multiple 
        window sizes (e.g., 50ms, 120ms, 240ms). This forces the network to accurately reconstruct 
        the spectral harmonics, which aligns with how the human ear actually perceives audio quality.
        """
        # 1. Time-Domain Loss (Waveform structure)
        l1_loss = F.l1_loss(denoised_audio, clean_audio)
        
        # 2. Frequency-Domain Loss (Human perceptual quality)
        # Spectral Convergence (sc):
        # Calculates the Frobenius norm (matrix norm) of the difference between the target and predicted magnitude spectrograms, divided by the norm of the target.
        # Log Magnitude (mag):
        # This calculates the L1 distance (absolute error) between the logarithm of the spectrograms.
        sc_loss, mag_loss = self.stft_loss(denoised_audio, clean_audio)
        stft_loss = sc_loss + mag_loss
        
        loss = l1_loss + stft_loss
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('l1_loss', l1_loss)
        self.log('stft_loss', stft_loss)
        return loss

    def configure_optimizers(self):
        """ Learning Rate
        The CleanUMamba authors specify using the ADAM optimizer with a learning rate of 0.0002.
        This is the community standard for State Space Models.
        Because Mamba relies on a continuous mathematical recurrence loop,
        larger learning rates (like 1e-3) often cause the gradients to explode, resulting in NaN losses.
        """
        """ Weight Decay Sorting
        Separate 1D and 2D parameters for weight decay.
        Why: Weight decay smooths convolutional filters (good), but decaying Mamba's 
        1D state parameters (A, B, C, Delta) artificially shrinks its time-constants 
        and destroys its long-term memory capacity.
        """
        decay_params = []
        no_decay_params = []
        
        for _, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            # If the parameter is 1D (biases, norm weights, Mamba state vectors)
            if param.ndim <= 1:
                no_decay_params.append(param)
            # If the parameter is 2D or higher (Convolutional filters, Linear weights)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.02},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(optim_groups, lr=2e-4)
