"""
Streaming Inference Mixin (Real-Time State Management)

OVERVIEW:
Standard PyTorch models are "stateless" during inference. If you pass a 10-second 
audio file through a U-Net, it calculates the padding and receptive fields for the 
entire file at once. 

In a live B2B communications environment (VoIP/Telehealth), you do not have the whole file. 
You receive continuous, tiny chunks of audio (e.g., 10-millisecond buffers). If you feed 
these chunks individually into a stateless U-Net, the convolutional layers and Mamba 
blocks will have no acoustic memory of the previous chunk, resulting in severe audio 
stuttering and phase distortion at the boundaries of every buffer.

PURPOSE:
This Mixin injects stateful acoustic memory into the PyTorch architecture without 
mangling the core mathematical graph used for training. 

MECHANICS:
1. Buffer Management (`self.pending`): It holds incoming microphone arrays until enough 
   samples have accumulated to perfectly divide through the U-Net's stride architecture.
2. The Overlap-Add Cache (`self.encoder_decoder_state`): It caches the trailing edges 
   of the 1D convolutions from Frame A, so they can be mathematically added to the 
   leading edges of Frame B.
3. The SSM Cache (`self.inference_params`): It captures Mamba's internal hidden state 
   vector (h_t) at the exact moment Frame A finishes processing, injecting it as the 
   starting state for Frame B.

By inheriting this Mixin, the PyTorch model mimics the exact C++ memory allocation 
strategy required for edge deployment, allowing us to test real-time streaming locally.
"""

import torch
import time

class StreamingInferenceMixin:
    # Fields from EdgeAudioUSE class
    num_encoder_layers: int
    stride: int
    frame_length: int

    """
    A Mixin class that adds real-time edge-streaming capabilities to an audio U-Net.
    Manages audio buffers, convolutional overlap-add states, and Mamba state caches.
    """
    def init_streaming_state(self, channels_input=1, dtype=torch.float32, device='cpu'):
        """Call this after initializing your model to prep the streaming buffers."""
        self.total_time = 0
        self.frames = 0
        self.input_std = 0
        self.pending = torch.zeros(channels_input, 0, dtype=dtype, device=device)
        self.inference_params = {}
        self.encoder_decoder_state = {}
        
        # Requires the model to have stride and num_encoder_layers defined
        self.total_stride_val = self.stride ** self.num_encoder_layers

    def reset_streaming_state(self):
        self.total_time = 0
        self.frames = 0
        self.pending = torch.zeros_like(self.pending[:, :0])
        self.inference_params = None
        self.encoder_decoder_state = {}

    def flush(self):
        """Flushes remaining audio by padding with zeros and returns the last chunk."""
        self.encoder_decoder_state = {}
        pending_length = self.pending.shape[1]
        
        # Requires get_valid_length to be available or calculated
        padding = torch.zeros(self.pending.shape[0], self.frame_length, 
                              device=self.pending.device, dtype=self.pending.dtype)
        out = self.feed(padding)
        return out[:, :pending_length]

    @torch.no_grad()
    def feed(self, noisy_input):
        """
        Ingests a raw stream of audio. If the buffer hits the valid frame length, 
        it pops the frame, runs inference, caches the state, and returns the clean audio.
        """
        if noisy_input.dim() != 2:
            raise ValueError("Input should be two dimensional (Channels, Time).")
        
        # Ensure the buffer follows the input to the correct hardware
        if self.pending.device != noisy_input.device:
            self.pending = self.pending.to(noisy_input.device)
            
        begin = time.time()
        self.pending = torch.cat([self.pending, noisy_input], dim=1)
        denoised_frames = []
        
        while self.pending.shape[1] >= self.frame_length:
            self.frames += 1
            frame = self.pending[:, :self.frame_length]

            # Run your custom step-by-step forward pass here
            out = self._enhance_frame_step(frame)
            
            out = out[:, :self.total_stride_val]
            denoised_frames.append(out)
            
            # Slide the pending buffer forward by the stride amount
            self.pending = self.pending[:, self.total_stride_val:]

        self.total_time += time.time() - begin

        if denoised_frames:
            return torch.cat(denoised_frames, 1)
        return torch.zeros(noisy_input.shape[0], 0, device=noisy_input.device)

    def _enhance_frame_step(self, frame):
        raise NotImplementedError("Model must implement _enhance_frame_step for streaming.")