# A step-by-step implementation of the MFCC feature extraction using various modules from PyTorch
import torch
import torchaudio
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# Load the audio file
waveform, sample_rate = torchaudio.load('BabyElephantWalk60.wav')

# Define hyperparameters

n_fft = 2048 
"""
In context of STFT, n_fft can be though of as frame length. The n_fft parameter defines the number of samples in each frame.
In the time-domain n_fft specifies how many audio samples are included in each frame, and in frequency domain it determines the resolution of the frequency bins 
after applying the FFT to each frame.
"""
hop_length = 512 # It specifies the number of samples between the start of two consecutive frames. 

n_mels = 128 
n_mfcc = 13 # Number of MFCCs to retain

# Apply pre-emphasis 
"""
It is used to amplify the high-frequency components of an audio signal before further analysis. It boosts the energy of the high-frequency components relative to the low-frequency components.
"""
pre_emphasis = 0.97 # pre_emphasis coefficient 

"""
torch.cat(tensors, dim = ,*,None Concatenates the given sequence of tensors in the given dimension.
waveform[:, 0:1] denotes the first sample, waveform[:, 1:] denotes all the samples from the secondd sample onwards, and, waveform[:, :-1] denotes all the samples except the last one.
dim = 1, means along the column dimension   
"""
waveform = torch.cat((waveform[:, 0:1], waveform[:, 1:] - pre_emphasis * waveform[:, :-1]), dim=1)


# Compute the Short-Time Fourier Transform (STFT)
# Computes the STFT providing a time-frequency representation of the audio signal including both the magnitude and phase information.
"""
Information of the parameters used:
torch.stft: Computes the Short-Time Fourier Transform of the input waveform. 
waveform: The input audio signal. 
n_fft: The number of FFT points (frame length). 
hop_length: The number of samples between adjacent STFT columns (frame shift). 
win_length: The length of the window function. 
window: The window function applied to each segment (Hamming window). 
return_complex: Specifies that the output should be a complex tensor.
"""
stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hamming_window(n_fft), return_complex=True)

"""
The power spectrum provides information about the energy of different frequency components in the signal.
"""
# Compute the power spectrum
power_spectrum = stft.abs() ** 2

"""
It creates a Mel filter bank using the 'MelScale' transformation from the torchaudio.transforms module. 
The Mel filter bank is used to convert the power spectrum of the audio signal into the Mel scale, which better represents how humans perceive sound.
It consists of a series of triangular filters spaced according to the Mel scale.
"""
# Create Mel filter bank
mel_filter_bank = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1)

"""
This line of code applies the Mel filter bank to the power spectrum to create the Mel spectrogram. 
The Mel spectrogram represents the energy of the audio signal in the Mel scale, which is more aligned with human auditory perception.
"""

# Apply the Mel filter bank to the power spectrum
mel_spectrogram = mel_filter_bank(power_spectrum)

# Take the logarithm of the Mel spectrogram
log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

# Compute the MFCCs by applying the Discrete Cosine Transform (DCT)

# Convert log Mel spectrogram from a PyTorch tensor to a NumPy array for DCT computation
# As the module scipy.fftpack.dct takes input as an array, instead of a tensor. So,the conversion is required.
log_mel_spectrogram_np = log_mel_spectrogram.numpy()

# Apply the Discrete Cosine Transform (DCT) using scipy.fftpack.dct
mfccs_np = dct(log_mel_spectrogram_np, type=2, axis=-2, norm='ortho')[:n_mfcc]

# Convert back to a PyTorch tensor
mfccs = torch.tensor(mfccs_np)

# Plot the MFCCs
plt.figure(figsize=(10, 6))
plt.imshow(mfccs[0].detach().numpy(), origin='lower', aspect='auto', cmap='viridis')
plt.title('MFCCs')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()