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
model = keras.models.load_model("my_model.h5")

def RecordAudio():
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
        return

def feature_extraction(List, path):
        # reading audio file
        X, sample_rate = librosa.load(path)
        Mcc = librosa.feature.mfcc(y=X, sr= sample_rate, n_mfcc=13)
        Mcc = np.mean(Mcc.T, axis = 0)
        chroma_stft = librosa.feature.chroma_stft(y=X,sr=sample_rate,n_chroma=12,n_fft=4096)
        chroma_stft = np.mean(chroma_stft.T, axis=0)
        spectral_bandwidth= librosa.feature.spectral_bandwidth(y=X, sr=sample_rate, n_fft=4096)
        spectral_bandwidth= np.mean(spectral_bandwidth.T, axis =0)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.percussive(X), sr=sample_rate)
        tonnetz = np.mean(tonnetz.T, axis = 0)
        melspectrogram = librosa.feature.melspectrogram(y=X,sr= sample_rate,n_fft = 4096)
        melspectrogram = np.mean(melspectrogram.T, axis=0)
        spectral_centroid = librosa.feature.spectral_centroid(y=X,sr=sample_rate,n_fft=4096)
        spectral_centroid = np.mean(spectral_centroid.T, axis= 0)
        spectral_contrast = librosa.feature.spectral_contrast(y=X,sr=sample_rate, n_fft=4096)
        spectral_contrast = np.mean(spectral_contrast.T, axis= 0)
        
        feature = np.hstack((Mcc,spectral_bandwidth,chroma_stft,melspectrogram,spectral_centroid,spectral_contrast,tonnetz))
        List.append(feature)
        return List
    
def Classify( args):
        
        Z = []
        feature_extraction(Z, args)
        Z_arr = np.array(Z)
        Z_arr = np.expand_dims(Z_arr, axis=2)
        prediction = model.predict(Z_arr, batch_size = num_batch_size)
        print(Z_arr)
        prediction = prediction.argmax(axis = 1)
        if (prediction == 0):
            return "other"
        elif (prediction == 1):
            return "parvez"
        else: 
            return "noise"

from kivymd.uix.button import MDFillRoundFlatButton
from kivymd.uix.screen import MDScreen
from kivymd.app import MDApp
from kivy.uix.image import Image
from kivymd.uix.button import MDFillRoundFlatIconButton, MDFillRoundFlatButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.label import MDLabel
#from kivy.uix.toolbar import MDToolbar
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.icon_definitions import md_icons
from numpy import source

class ConverterApp(MDApp):
    
   
    def flip(self):
        print("working..")
    def recording(self,args):
        self.result.text = "recording started"
        
        print("recording stared")
        RecordAudio()
        self.result.text = "recording ended"
        print("recording ended")
        return
    def testing(self,args):
        speaker = Classify("testing.wav")
        self.result.text = speaker

    def build(self):
        screen = MDScreen(md_bg_color = "E8E8E3", size_hint=(0.7,0.9),pos_hint ={"center_x":0.5, "center_y":0.5})
        self.theme_cls.primary_palette = "Teal"
        self.toolbar = MDTopAppBar(title = "Voice Authentication")
        self.toolbar.pos_hint = {"top":1}
        self.toolbar.right_action_items=[["dots-vertical-circle",lambda x: self.flip()]]
        screen.add_widget(self.toolbar)

        self.logo = Image(source = "iba_logo.jpg",
                            pos_hint = {"center_x":0.5,"center_y":0.7},
                            size_hint = (0.3,0.3))
        screen.add_widget(self.logo)
        self.label = MDLabel(text= "Welcome to vocie authentication!",
                             font_size = 20,
                              theme_text_color = "Primary",
                              background = "#008080",
                              text_color = "#008080",
                              color= "#008080",
                              halign = "center",
                              pos_hint = {"center_x":0.5,"center_y":0.5}
                              )
        screen.add_widget(self.label)
        self.result = MDLabel(
                              font_size = 20,
                              theme_text_color = "Primary",
                              background = "#008080",
                              text_color = "#008080",
                              color = "#008080",
                              halign = "center",
                              pos_hint = {"center_x":0.5,"center_y":0.45}
            
        )
        screen.add_widget(self.result)
        self.record = MDFillRoundFlatButton(text = "Record",
                            pos_hint = {"center_x":0.5,"center_y":0.38},
                            width = 200,
                            halign = "center",
                            height= 30,
                            md_bg_color = "#008080",
                            font_size= 20,
                            on_press = self.recording
                            )
        self.test = MDFillRoundFlatButton(text = "Predict",
                            pos_hint = {"center_x":0.5,"center_y":0.3},
                            width = 200,
                            halign= "center",
                            height= 30,
                            md_bg_color = "#008080",
                            font_size= 20,
                            on_press = self.testing
                            )
        screen.add_widget(self.record)
        screen.add_widget(self.test)
        return screen

if __name__ == '__main__':
    ConverterApp().run()