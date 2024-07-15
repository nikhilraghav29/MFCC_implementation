# An implementation of the MFCC feature extraction using the MFCC module from torchaudio.transforms

import torch
import torchaudio
import matplotlib.pyplot as plt

# Load the audio file
waveform, sample_rate = torchaudio.load('BabyElephantWalk60.wav')

# Define the parameters for MFCC
n_mfcc = 13
melkwargs = {
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 128,
    'center': True,
    'pad_mode': 'reflect',
    'mel_scale': 'htk'
}

# Apply the MFCC transformation
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs=melkwargs
)

# Extract MFCC features
mfcc = mfcc_transform(waveform)

# Convert MFCC tensor to numpy array for visualization
mfcc_np = mfcc.squeeze().numpy()

# Visualize the MFCCs
plt.figure(figsize=(10, 6))
plt.imshow(mfcc_np, cmap='viridis', origin='lower', aspect='auto')
plt.title('MFCCs')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.colorbar(format='%+2.0f dB')
plt.show()
