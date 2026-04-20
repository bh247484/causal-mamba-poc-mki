# simulate_streaming.py
import torch
import torchaudio
import torch.nn.functional as F
from model import EdgeAudioUSE

def main():
    print("--- Phase 1: Mathematical Verification (Python) ---")

    # 1. Load the Model
    print("Loading EdgeAudioUSE architecture...")
    model = EdgeAudioUSE()

    # 1. Point to your specific version_12 checkpoint
    # You will need to check your tb_logs folder for the exact filename (e.g., 'epoch=499-step=1500.ckpt')
    ckpt_path = "tb_logs/StageA_Overfit/version_12/checkpoints/use-overfit-epoch=496-train_loss=0.2653.ckpt"
    
    print(f"Loading weights from {ckpt_path}...")
    
    # 2. Load the Lightning archive into CPU memory
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    # 3. Extract just the weights dictionary
    state_dict = checkpoint['state_dict']
    
    # 4. Strip the PyTorch Lightning prefix (if it exists)
    # If your training script used a wrapper like `self.model = EdgeAudioUSE()`, 
    # all keys will start with 'model.'. We must remove that so the names match.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Change 'model.' to whatever prefix PL added, or leave it if there isn't one
        clean_key = key.replace("model.", "") 
        cleaned_state_dict[clean_key] = value

    # 5. Load the cleaned weights into your architecture
    # strict=True ensures it crashes if we missed a layer or prefix
    model.load_state_dict(cleaned_state_dict, strict=True)
    
    # CRITICAL: Disable training behaviors (like gradient tracking)
    model.eval() 

    # 2. Load the Audio
    # Replace this with the actual path to the noisy file you used to overfit
    audio_path = "datasets/VoiceBank-DEMAND/noisy_trainset_28spk_wav/p259_330.wav" 
    
    try:
        print(f"Loading audio from {audio_path}...")
        noisy_wav, sr = torchaudio.load(audio_path)
        # Ensure it's 16kHz
        if sr != 16000:
            noisy_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(noisy_wav)
            
        # Trim or pad to exactly 3.0 seconds (48,000 samples) to match your dataset loader
        current_length = noisy_wav.shape[-1]
        if current_length > 48000:
            noisy_wav = noisy_wav[:, :48000]
        elif current_length < 48000:
            pad_amount = 48000 - current_length
            noisy_wav = F.pad(noisy_wav, (0, pad_amount))

    except Exception as e:
        print(f"Could not load {audio_path}. Generating dummy 3.0s audio for math verification.")
        # Fallback to random noise so the math test can run regardless
        noisy_wav = torch.randn(1, 48000)

    # ---------------------------------------------------------
    # PASS A: Stateless (The Training Graph)
    # ---------------------------------------------------------
    print("Executing Pass A: Stateless forward()...")
    # forward() expects (Batch, Channels, Time)
    stateless_input = noisy_wav.unsqueeze(0) 
    
    with torch.no_grad():
        output_A = model.forward(stateless_input)
    
    # Remove batch and channel dims -> (Time,) for easy comparison
    output_A = output_A.squeeze() 

    # ---------------------------------------------------------
    # PASS B: Stateful (The Edge Inference Graph)
    # ---------------------------------------------------------
    print("Executing Pass B: Stateful feed() loop...")
    model.init_streaming_state(channels_input=1)
    
    chunk_size = 160  # 10ms at 16kHz
    output_chunks = []
    
    with torch.no_grad():
        # Iterate over the audio in exact 160-sample chunks
        for i in range(0, noisy_wav.shape[-1], chunk_size):
            # feed() expects (Channels, Time) -> (1, chunk_size)
            chunk = noisy_wav[:, i:i+chunk_size]
            chunk_out = model.feed(chunk)
            
            if chunk_out.shape[-1] > 0:
                output_chunks.append(chunk_out)
                
        # Flush any remaining internal buffers padded with zeros
        flushed_out = model.flush()
        if flushed_out.shape[-1] > 0:
            output_chunks.append(flushed_out)

    # Concatenate all streamed chunks and remove channel dim -> (Time,)
    output_B = torch.cat(output_chunks, dim=-1).squeeze()

    # ---------------------------------------------------------
    # THE ASSERTION
    # ---------------------------------------------------------
    print("\nComparing outputs...")
    
    # Trim to minimum length
    # Due to padding and the flush() command, the stateful pass might append 
    # a few extra zeros at the very tail end. We align the lengths here.
    min_len = min(output_A.shape[-1], output_B.shape[-1])
    output_A_trimmed = output_A[:min_len]
    output_B_trimmed = output_B[:min_len]

    print("\nSaving audio files for perceptual comparison...")
    
    # torchaudio expects (Channels, Time), so we unsqueeze to add the mono channel back
    # and ensure they are on the CPU
    audio_A = output_A_trimmed.unsqueeze(0).cpu()
    audio_B = output_B_trimmed.unsqueeze(0).cpu()
    
    # The Residual: This is the literal sound of the math failing
    residual = (audio_A - audio_B)
    
    torchaudio.save("pass_A_stateless.wav", audio_A, 16000)
    torchaudio.save("pass_B_stateful.wav", audio_B, 16000)
    torchaudio.save("bug_residual.wav", residual, 16000)
    
    print("Files saved to disk:")
    print(" - pass_A_stateless.wav")
    print(" - pass_B_stateful.wav")
    print(" - bug_residual.wav (Listen to this first!)")

    # Calculate Mean Squared Error and Max Absolute Error
    mse = F.mse_loss(output_A_trimmed, output_B_trimmed).item()
    max_diff = torch.max(torch.abs(output_A_trimmed - output_B_trimmed)).item()

    print(f"Mean Squared Error (MSE):  {mse:.10f}")
    print(f"Max Absolute Difference:   {max_diff:.10f}")

    if mse < 1e-6:
        print("\n✅ SUCCESS: Streaming logic is mathematically identical to training logic.")
        print("Your overlap-add caches and Mamba state passing are flawless.")
        print("You are officially cleared to begin writing the ggml dump scripts.")
    else:
        print("\n❌ FAILURE: State caching bug detected in _enhance_frame_step.")
        print("The arrays diverge. Do not export to C++ until this MSE is 0.0.")

if __name__ == "__main__":
    main()