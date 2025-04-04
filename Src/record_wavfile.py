# python -m pip install pyaudio
from statistics import mode
import pyaudio
import librosa
import wave
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

num_batch_size = 32

pa = pyaudio.PyAudio()

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

print('start recording')

seconds = 2
frames = []
second_tracking = 0
second_count = 0
for i in range(0, int(RATE/FRAMES_PER_BUFFER*seconds)):
    data = stream.read(FRAMES_PER_BUFFER)
    frames.append(data)
    second_tracking += 1
    if second_tracking == RATE/FRAMES_PER_BUFFER:
        second_count += 1
        second_tracking = 0
        print(f'Time Left: {seconds - second_count} seconds')


stream.stop_stream()
stream.close()
pa.terminate()

obj = wave.open('testing.wav', 'wb')
obj.setnchannels(CHANNELS)
obj.setsampwidth(pa.get_sample_size(FORMAT))
obj.setframerate(RATE)
obj.writeframes(b''.join(frames))
obj.close()


file = wave.open('testing.wav', 'rb')

# sample_freq = file.getframerate()
# frames = file.getnframes()
# signal_wave = file.readframes(-1)

# file.close()

# time = frames / sample_freq


# # if one channel use int16, if 2 use int32
# audio_array = np.frombuffer(signal_wave, dtype=np.int16)

# times = np.linspace(0, time, num=frames)

# plt.figure(figsize=(15, 5))
# plt.plot(times, audio_array)
# plt.ylabel('Signal Wave')
# plt.xlabel('Time (s)')
# plt.xlim(0, time)
# plt.title('The Thing I Just Recorded!!')
# plt.show()




def feature_extraction(List, path):
    # reading audio file
    X, sample_rate = librosa.load(path)
    
    # MFCC extraction
    Mcc = librosa.feature.mfcc(y=X, sr= sample_rate, n_mfcc=13)
    Mcc = np.mean(Mcc.T, axis = 0)
    
    # chroma_stft extraction
    chroma_stft = librosa.feature.chroma_stft(y=X,sr=sample_rate,n_chroma=12,n_fft=4096)
    chroma_stft = np.mean(chroma_stft.T, axis=0)
    
    # chroma_cqt extraction
    #chroma_cqt = librosa.feature.chroma_cqt(y=X, sr=sample_rate)
    #chroma_cqt = np.mean(chroma_cqt.T,axis = 0)
    
    #spectral bandwidth extraction
    spectral_bandwidth= librosa.feature.spectral_bandwidth(y=X, sr=sample_rate, n_fft=4096)
    spectral_bandwidth= np.mean(spectral_bandwidth.T, axis =0)
    
    # spectral flattness
    #flatness = librosa.feature.spectral_flatness(y = X, n_fft = 4096)
    #flatness = np.mean(flatness.T , axis=0)
    
    # tonnetz extraction
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.percussive(X), sr=sample_rate)
    tonnetz = np.mean(tonnetz.T, axis = 0)
    
    # melspectrogram
    melspectrogram = librosa.feature.melspectrogram(y=X,sr= sample_rate,n_fft = 4096)
    melspectrogram = np.mean(melspectrogram.T, axis=0)
    
    #spectral centroid extraction
    spectral_centroid = librosa.feature.spectral_centroid(y=X,sr=sample_rate,n_fft=4096)
    spectral_centroid = np.mean(spectral_centroid.T, axis= 0)
    
    # spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=X,sr=sample_rate, n_fft=4096)
    spectral_contrast = np.mean(spectral_contrast.T, axis= 0)
    
    feature = np.hstack((Mcc,spectral_bandwidth,chroma_stft,melspectrogram,spectral_centroid,spectral_contrast,tonnetz))
    List.append(feature)
    return List
    

model = keras.models.load_model("my_model.h5")

def classify(path):
    Z = []
    feature_extraction(Z, path)
    Z_arr = np.array(Z)
    Z_arr = np.expand_dims(Z_arr, axis=2)
    prediction = model.predict(Z_arr, batch_size = num_batch_size)
    print(Z_arr)
    prediction = prediction.argmax(axis = 1)
    if (prediction == 0):
        return "Other"
    elif (prediction == 1):
        return "Parvez"
    else: 
        return "Noise"

print(classify('manoj_test_same.wav'))



