from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

n_mels = 64
hop_length_ms = 16.0
sample_rate = 16000
n_bins = 8

def audio_to_logmel_spectrogram(audio, n_mels, hop_length_ms, sample_rate):
    hop_length = int(sample_rate * hop_length_ms / 1000.0)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=2048,
        fmin=0,
        fmax=sample_rate/2
    )
    
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    min_val = np.min(log_mel)
    max_val = np.max(log_mel)
    
    return log_mel, min_val, max_val

def quantize_spectrogram(spectrogram, n_bins):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
   
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    quantized_spectrogram = np.digitize(spectrogram, bins)
    return quantized_spectrogram

def discrete_amplitude_derivative_matrix(quantized_spectrogram):
    quantized_spectrogram_diff = np.diff(quantized_spectrogram, axis=1)
    return quantized_spectrogram_diff

def clip_derivative_matrix(derivative_matrix):
    derivative_matrix = np.clip(derivative_matrix, -1, 1)
    return derivative_matrix

def derivative_matrix_to_quantized(derivative_matrix, n_bins):
    n_mels, n_frames = derivative_matrix.shape
    initial_frame = np.zeros((n_mels, 1), dtype=derivative_matrix.dtype)
    
    quantized_spectrogram = np.cumsum(derivative_matrix, axis=1)
    quantized_spectrogram = np.concatenate([initial_frame, quantized_spectrogram], axis=1)
    
    quantized_clamped = np.clip(quantized_spectrogram, 1, n_bins+1)
    
    return quantized_clamped

def dequantize_spectrogram(quantized_spectrogram, n_bins, min_val, max_val):
    bins = np.linspace(min_val, max_val, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    log_mel_spectrogram = bin_centers[quantized_spectrogram - 1]
    
    return log_mel_spectrogram

def logmel_spectrogram_to_audio(log_mel_spectrogram, hop_length_ms, sample_rate, n_mels, n_iter=32):
    hop_length = int(sample_rate * hop_length_ms / 1000.0)
    
    mel_spec = librosa.db_to_power(log_mel_spectrogram, ref=1.0)
    
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=2048,
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate/2
    )
    
    #pseudo-inverse of the mel basis matrix applied to the mel spectrogram to get the magnitude spectrogram
    magnitude_spec = np.dot(np.linalg.pinv(mel_basis), mel_spec)
    
    #Griffin-Lim starts with random phase and converts to 1D audio. Does STFT, keeps new phase and discards new magnitude. Does iteratively until convergence.
    audio = librosa.griffinlim(
        magnitude_spec,
        hop_length=hop_length,
        n_iter=n_iter,
        n_fft=2048
    )
    
    return audio

def derivative_matrix_to_audio(derivative_matrix, n_bins, hop_length_ms, sample_rate, n_mels, 
                               min_val, max_val, n_iter=32):
    quantized_spectrogram = derivative_matrix_to_quantized(derivative_matrix, n_bins)
    log_mel_spectrogram = dequantize_spectrogram(quantized_spectrogram, n_bins, min_val, max_val)
    audio = logmel_spectrogram_to_audio(log_mel_spectrogram, hop_length_ms, sample_rate, n_mels, n_iter)
    
    #rescale audio to -0.9 to 0.9
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude > 0:
        audio = audio / max_amplitude
    
    return audio

def visualize_quantized_and_matrix(quantized_spectrogram, quantized_clamped, derivative_matrix, title):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    ax1 = axes[0]
    im1 = ax1.imshow(quantized_spectrogram, aspect='auto', origin='lower', cmap='inferno', interpolation='nearest')
    ax1.set_title('Quantized Spectrogram', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Frame', fontsize=12)
    ax1.set_ylabel('Mel Bin', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Bin Index', fraction=0.05, pad=0.05)
    
    ax2 = axes[1]
    im2 = ax2.imshow(derivative_matrix, aspect='auto', origin='lower', cmap='coolwarm', interpolation='nearest')
    ax2.set_title('Derivative Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Frame', fontsize=12)
    ax2.set_ylabel('Mel Bin', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Derivative Value', fraction=0.05, pad=0.05)
    
    ax3 = axes[2]
    im3 = ax3.imshow(quantized_clamped, aspect='auto', origin='lower', cmap='inferno', interpolation='nearest')
    ax3.set_title('Reconstructed Quantized Spectrogram', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Frame', fontsize=12)
    ax3.set_ylabel('Mel Bin', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='Bin Index', fraction=0.05, pad=0.05)

    n_mels, n_frames = derivative_matrix.shape
    
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1)
    plt.tight_layout()
    plt.show()

def main():
    from pathlib import Path
    import random
    
    data_dir = Path('Speech_Commands')
    test_folders = ['dog', 'cat', 'yes', 'no', 'bed']
    test_file = None
    
    available_folders = []
    for folder_name in test_folders:
        folder = data_dir / folder_name
        if folder.exists():
            wav_files = list(folder.glob('*.wav'))
            if wav_files:
                available_folders.append((folder_name, wav_files))
    
    if available_folders:
        folder_name, wav_files = random.choice(available_folders)
        test_file = random.choice(wav_files)
    
    audio, sr = librosa.load(test_file)
    spectrogram, min_val, max_val = audio_to_logmel_spectrogram(audio, n_mels, hop_length_ms, sample_rate)
    quantized_spectrogram = quantize_spectrogram(spectrogram, n_bins)
    intermediate_matrix = discrete_amplitude_derivative_matrix(quantized_spectrogram)
    derivative_matrix = clip_derivative_matrix(intermediate_matrix)
    quantized_clamped = derivative_matrix_to_quantized(derivative_matrix, n_bins)
    audio_from_derivative = derivative_matrix_to_audio(derivative_matrix, n_bins, hop_length_ms, sample_rate, n_mels, 
                                                       min_val=min_val, max_val=max_val)
    
    print(f"Min value: {min_val}, Max value: {max_val}")
    visualize_quantized_and_matrix(
        quantized_spectrogram, 
        quantized_clamped,
        derivative_matrix,
        title=f"Quantized Spectrogram and Derivative Matrix - {test_file.name}"
    )
    
    import soundfile as sf
    reconstructed_file = Path(__file__).parent / 'reconstructed_audio.wav'
    sf.write(str(reconstructed_file), audio_from_derivative, sample_rate)
    print(f"Reconstructed audio saved to {reconstructed_file.absolute()}")
    
    original_file = Path(__file__).parent / 'original_audio.wav'
    sf.write(str(original_file), audio, sample_rate)
    print(f"Original audio saved to {original_file.absolute()}")

if __name__ == '__main__':
    main()
