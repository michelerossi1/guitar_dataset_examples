

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os




folder_path = 'audio/effected'

for root, dirs, files in os.walk(folder_path):
    
    for file in files:
        
        full_path = os.path.join(root, file)
        
        signal, sr = librosa.load(full_path, sr=22050)
        
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        # Plot and save the log-Mel spectrogram
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format="%+2.0f dB")
        plt.title('Log-Mel Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel Frequency (Hz)')
        
        
        output_path = os.path.join('images', file[:-4] + '.png')
        plt.savefig(output_path)
        plt.close()



import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

folder_path = 'audio/effected'

# Set global font size for matplotlib
plt.rcParams.update({
    'font.size': 16,         # Default font size for all text
    'axes.titlesize': 20,    # Font size for titles
    'axes.labelsize': 18,    # Font size for axis labels
    'xtick.labelsize': 14,   # Font size for x-axis ticks
    'ytick.labelsize': 14,   # Font size for y-axis ticks
    'text.color': 'gray',    # Default text color
    'axes.labelcolor': 'gray',  # Axis labels color
    'xtick.color': 'gray',   # X-axis tick color
    'ytick.color': 'gray',   # Y-axis tick color
})
for root, dirs, files in os.walk(folder_path):
    for file in files:
        full_path = os.path.join(root, file)
        
        # Load audio file
        signal, sr = librosa.load(full_path, sr=22050)
        
        # Generate log-Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        # Plot and save the log-Mel spectrogram with larger font
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        cbar = plt.colorbar(format="%+2.0f dB")
        cbar.ax.tick_params(labelsize=14)  # Adjust colorbar tick font size
        plt.title('Log-Mel Spectrogram', fontsize=20)  # Larger title font
        plt.xlabel('Time (s)', fontsize=18)  # Larger x-axis label font
        plt.ylabel('Mel Frequency (Hz)', fontsize=18)  # Larger y-axis label font
        
        # Save the spectrogram
        output_path = os.path.join('images', file[:-4] + '.png')
        plt.savefig(output_path, bbox_inches='tight')  # Save without extra whitespace
        plt.close()
