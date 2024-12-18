

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



