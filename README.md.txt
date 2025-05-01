# Audio Classification Project

This project records audio, extracts features from the recorded audio, and classifies it using a CNN deep learning model (`my_model.h5`). The classification model predicts whether the audio belongs to one of three categories: "other", "parvez", or "noise". The user interface is built using `Kivy` and `KivyMD`.

## Features

- Audio recording using `pyaudio`
- Audio feature extraction using `librosa`
- Classification with a pre-trained TensorFlow/Keras model
- User interface built with `Kivy` and `KivyMD`

## Requirements

You need to install the following dependencies:

- `pyaudio` for audio recording
- `librosa` for audio feature extraction
- `matplotlib` for visualizations (optional)
- `numpy` for numerical computations
- `tensorflow` for machine learning model predictions
- `kivy` and `kivymd` for the user interface

### Installing Dependencies

1. Create a virtual environment (optional, but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## How to Use

### 1. Record Audio

To record audio, you can run the `RecordAudio()` function. This will record audio for 2 seconds and save it as a `.wav` file (`testing.wav`).

### 2. Feature Extraction

Once the audio is recorded, the `feature_extraction()` function extracts various audio features, such as MFCCs, spectral bandwidth, chroma, and more.

### 3. Classification

Once features are extracted, the `Classify()` function will use the pre-trained model (`my_model.h5`) to predict the class of the recorded audio. It will print the predicted label: "other", "parvez", or "noise".

### 4. Running the Application

To run the full application with the user interface, execute the `MyApp.py` file from the `source` folder. This file provides a Kivy-based interface to interact with the application.

#### Running the Kivy Application

1. Ensure your virtual environment is activated and dependencies are installed.
2. Run the `MyApp.py` file:

    ```bash
    python source/MyApp.py
    ```

    This will start the Kivy application, allowing you to interact with the audio recording, feature extraction, and classification process directly through the UI.

