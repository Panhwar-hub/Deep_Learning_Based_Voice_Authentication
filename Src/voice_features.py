import librosa
import numpy as np


def feature_extraction(path):
    List = []
    # reading audio file
    X, sample_rate = librosa.load(path)

    # MFCC extraction
    Mcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    Mcc = np.mean(Mcc.T, axis=0)

    # chroma_stft extraction
    chroma_stft = librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=12, n_fft=4096)
    chroma_stft = np.mean(chroma_stft.T, axis=0)

    # chroma_cqt extraction
    # chroma_cqt = librosa.feature.chroma_cqt(y=X, sr=sample_rate)
    # chroma_cqt = np.mean(chroma_cqt.T,axis = 0)

    # spectral bandwidth extraction
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=X, sr=sample_rate, n_fft=4096)
    spectral_bandwidth = np.mean(spectral_bandwidth.T, axis=0)

    # spectral flattness
    # flatness = librosa.feature.spectral_flatness(y = X, n_fft = 4096)
    # flatness = np.mean(flatness.T , axis=0)

    # tonnetz extraction
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.percussive(X), sr=sample_rate)
    tonnetz = np.mean(tonnetz.T, axis=0)

    # melspectrogram
    melspectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=4096)
    melspectrogram = np.mean(melspectrogram.T, axis=0)

    # spectral centroid extraction
    spectral_centroid = librosa.feature.spectral_centroid(y=X, sr=sample_rate, n_fft=4096)
    spectral_centroid = np.mean(spectral_centroid.T, axis=0)

    # spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate, n_fft=4096)
    spectral_contrast = np.mean(spectral_contrast.T, axis=0)

    feature = np.hstack(
        (Mcc, spectral_bandwidth, chroma_stft, melspectrogram, spectral_centroid, spectral_contrast, tonnetz))
    List.append(feature)
    List = np.array(List)
    List = np.expand_dims(List, axis=2)
    return List


def main():
    A = feature_extraction("parvez_test_diff.wav")
    print(A.shape)


if __name__ == "__main__":
    main()