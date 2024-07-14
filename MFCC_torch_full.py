import torch
import torchaudio
import matplotlib.pyplot as plt

# Load the audio file
waveform, sample_rate = torchaudio.load('BabyElephantWalk60.wav')

# Define hyperparameters
n_fft = 2048
hop_length = 512
n_mels = 128
n_mfcc = 13

# Apply pre-emphasis
pre_emphasis = 0.97
waveform = torch.cat((waveform[:, 0:1], waveform[:, 1:] - pre_emphasis * waveform[:, :-1]), dim=1)

# Compute the Short-Time Fourier Transform (STFT)
stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hamming_window(n_fft), return_complex=True)

# Compute the power spectrum
power_spectrum = stft.abs() ** 2

# Create Mel filter bank
mel_filter_bank = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1)

# Apply the Mel filter bank to the power spectrum
mel_spectrogram = mel_filter_bank(power_spectrum)

# Take the logarithm of the Mel spectrogram
log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

# Compute the MFCCs by applying the Discrete Cosine Transform (DCT)
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': n_mels, 'center': False}
)

mfccs = mfcc_transform(waveform)

# Plot the MFCCs
plt.figure(figsize=(10, 6))
plt.imshow(mfccs[0].detach().numpy(), origin='lower', aspect='auto', cmap='viridis')
plt.title('MFCCs')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()