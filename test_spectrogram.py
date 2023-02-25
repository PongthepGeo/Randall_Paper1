import librosa
import librosa.display
# import IPython.display as ipd
import matplotlib.pyplot as plt

from playsound import playsound
# playsound('0_jackson_8.wav')

# scale_file = '0_jackson_8.wav'

# scale_file = '0_jackson_8.wav'
scale_file = '8_jackson_4.wav'

# load audio files with librosa
scale, sr = librosa.load(scale_file)
filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
print(filter_banks.shape)

# plt.figure(figsize=(25, 10))
# librosa.display.specshow(filter_banks, 
#                          sr=sr, 
#                          x_axis="linear")
# plt.colorbar(format="%+2.f")
# plt.show()

mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)

print(mel_spectrogram.shape)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

print(log_mel_spectrogram.shape)

plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
