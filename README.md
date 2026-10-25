# Causal Mamba PoC

## What is this repo?

This repo contains proof of concept code for a UNet Mamba Universal Speech Enhancement (USE) audio ML model as well as a C++/ggml real time inference engine designed to run that model on edge devices with resource constrained CPUs. This is a public artifact. I've since forked this code and continued work on a new, successor model.

The model is an attempt to prove real time, edge inference viability for an audio model architecture class that has so far largely been non causal and academic/experimental. It's an attempt to operationalize research.

If you're interested in the code, checkout to the following dirs: `/inference` and `/model`.

#### The `/model` dir contains...
- A pytorch lightning implementation of a causal UNet Mamba model with some custom layers/blocks and other useful neural net abstractions.
- A script to train/overfit the model.
- Script to export a custom binary file which encodes the various model layers and their weights.
  - The binary file is ultimately imported by the C++/ggml inference engine.
- Script to simulate streaming audio inference.
- Utility debugging scripts (per layer golden tensor style logging).

#### The `/inference` dir contains...
- A `model` which allocates memory and loads static weights per layer from the custom binary file described above.
- A `state` cache which preserves and forwards temporal context across streaming audio chunks.
- A ggml `graph` which contains a custom ggml implementation of the Unet Mamba model with layers and abstractions including...
  - Mamba blocks w/ ssm logic
  - Custom swiglu activation function
  - Skip connection and Unet bottle neck logic
- A command line app (`main`) used to test and debug ggml arena memory allocation and model binary loading.
- Another command line app (`benchmark`) used to test real time audio functionality and capture RTF (Real Time Factor) metrics.

## Motivations, why did I build this?

Partly to satisfy my curiosity and partly because of a strong conviction/intuition that there are opportunities to disrupt incumbent businesses built on traditional ML cloud service pricing (i.e. dollars per hour audio processed) with edge models. If machine learning workloads are distributed on edge devices (your customer's devices) then your business scales with zero marginal costs whereas cloud based incumbents costs go up the more their customers use their products. The challenge then is operationalizing SOTA research (like ssm mamba models) to run on edge devices in order to compete against existing cloud products/businesses. And crucially to do so before that tech becomes widely adopted/standardized and absorbed into some other layer of the stack on edge devices, like in an operating system audio ML api or even in a dsp module on an embedded device.

## Novel Learnings

During early overfitting training runs I found the model/optimzer was "cheating" by preferring skip connections over mamba blocks to minimize loss. This is typical behavior and a known problem in Unet mamba architecture, particularly in short training runs (my overfitting runs were only 500 epochs). Traditionally the problem is solved with a teacher student distillation approach where a teacher model encourages/forces a student model to use mamba blocks over skip connections. I was not in a position to train a large teacher model. So I tried a simple but surprisingly effective alternative.

I created a scalar called `skip_gate` which I used to tune skip connection utilization and then made that scalar/gate a learned parameter so the model itself could optimize the gate ratio as it learned. I initialized the parameter to near zero so the model was forced to use the mamba block bottleneck to reconstruct the signal during early epochs. Then gradually (limited by learning rate) the skip gate could/would open up and allow some signal to travel across the skip connections and to help fill in the high frequency data the bottleneck encoder blocks compressed away.

Anecdotally, this worked quite well. Much better results on overfitting training runs with the learned skip gate in place. Eventually I'd like to do more thoughtful and controlled ablations and A/B testing to measure my findings more empirically. For now here's some tensorboard screenshots of a few crude metrics I captured...

#### L1 Loss
<img width="490" height="458" alt="Screenshot 2026-10-24 at 10 05 29 AM" src="https://github.com/user-attachments/assets/c743be6b-184b-48b8-b078-355015c8c19b" />

#### STFT Loss
<img width="485" height="474" alt="Screenshot 2026-10-24 at 10 06 03 AM" src="https://github.com/user-attachments/assets/552f7363-66ff-41d3-89d7-ada6be375f77" />

#### Mamba Grad
<img width="491" height="404" alt="Screenshot 2026-10-24 at 10 05 43 AM" src="https://github.com/user-attachments/assets/990df8bb-7c0c-482b-910a-9c524b0cb26b" />

The two training runs without the learned skip gate are `version_6` and `version_9` (the gray and yellow lines) and `version_19` (the blue line) has the learned skip gate applied. Introducing the learned skip gate improved both the L1 loss and the STFT loss dramatically. The `mamba_grad_norm` is measured by collecting all mamba block gradients and then L2 normalizing them. Of course I should double back and capture `mamba_grad_norm` measurements for those earlier training runs without the learned parameter to compare exactly how much utilization improved but I haven't had the chance.

Here's the predicted audio output from each model checkpoint. You'll notice without the learned parameter the predicted output audio much more closely resembles the noisy input (background chatter/voices present) rather than the clean target audio. And the predicted output with the learned skip gate applied eliminates the background chatter entirely, although the audio still contains some high frequency artifacts (high pitch whine/buzz).

#### Without learned skip gate (version_9)
https://github.com/user-attachments/files/33929739/version_9_predicted.wav

#### With learned skip gate (version_19)
https://github.com/user-attachments/files/33929753/version_19_predicted.wav

## What's next for this code/project?

I'm leaving this code here and publically visible as a checkpoint. Although the architecture is mathematically sound the code in this repo is not viable for edge/cpu inference. I discovered a memory bandwidth bottleneck in the model size itself. Although the model is a reasonable size to load into RAM on an edge device it does not reasonably fit into a cpu cache. The extra cpu cycles required to rotate weights out of RAM and into the cpu cache make real time inference impossible. At the model's current size compute that should be reserved for inference and audio thread calculations is instead used to juggle memory and move data between addresses.

So the plan is to rearchitect the model with a <10mb target size by shaving off some layers/dimensions (reduced `d_model`) and retraining on mel spectrograms rather than raw audio (mel spectrograms improve performance over raw audio on smaller models). I'll likely include a few other optimizations from this [RT-SEMamba paper](https://arxiv.org/abs/2608.12099) which was published in August 2026, a few months after I had already begun work on this project. Encouraging to know that other researchers are pursuing this line of inquiry (i.e. real time, causal mamba audio models).

## Thank Yous

I want to thank the authors of the [CleanUMamba](https://arxiv.org/abs/2410.11062) and [USEMamba](https://arxiv.org/abs/2505.21198) papers whose research inspired and informed this project.

I also want to thank the maintainers of and contributors to [miniaudio](https://github.com/mackron/miniaudio) a single header, cross platform audio library I've used in this and many other projects.
