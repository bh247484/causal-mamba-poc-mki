import torch
import time

class StreamingInferenceMixin:
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
        self.inference_params = None
        self.encoder_decoder_state = {}
        
        # Requires the model to have stride and encoder_layers defined
        self.total_stride_val = self.stride ** self.encoder_layers

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
            
        begin = time.time()
        self.pending = torch.cat([self.pending, noisy_input], dim=1)
        denoised_frames = []
        
        while self.pending.shape[1] >= self.frame_length:
            self.frames += 1
            frame = self.pending[:, :self.frame_length]

            # Run your custom step-by-step forward pass here
            out = self._denoise_frame_step(frame)
            
            out = out[:, :self.total_stride_val]
            denoised_frames.append(out)
            
            # Slide the pending buffer forward by the stride amount
            self.pending = self.pending[:, self.total_stride_val:]

        self.total_time += time.time() - begin

        if denoised_frames:
            return torch.cat(denoised_frames, 1)
        return torch.zeros(noisy_input.shape[0], 0, device=noisy_input.device)

    def _denoise_frame_step(self, frame):
        raise NotImplementedError("Model must implement _denoise_frame_step for streaming.")