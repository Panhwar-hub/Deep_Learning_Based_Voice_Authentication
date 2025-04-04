from tkinter import Image
from turtle import screensize
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout

class FYPapp(App):
    def build(self):
        
        self.window = GridLayout()
        self.window.cols = 1;
        self.window.size_hint = (0.6,0.7)
        self.window.pos_hint = {"center_x":0.5, "center_y":0.5}

        self.window.add_widget(Image(source="iba_logo.jpg"))
        self.greeting = Label(text= "Hi Welcome to Voice Authentication!",
                              font_size = 20,
                              color = '#00FFCE'
                                )

        self.window.add_widget(self.greeting)
        self.record = Button (text = "Record",
                             size_hint = (None,None),
                            bold = True,
                            width = 200,
                            height= 30,
                            background_color = "#00FFCE",
                            font_size= 20,
                            pos = (self.window.x/2, self.window.height/2)
                            )
        self.test = Button(text = "Test",
                            size_hint = (None,None),
                            bold = True,
                            width = 200,
                            height= 30,
                            font_size= 20,
                            background_color = "#00FFCE")
        self.record.pos_hint = {"center_x":0.5, "center_y":0.5}
        self.test.pos_hint = {"center_x":0.5, "center_y":0.5}
        self.record.bind(on_press = self.Record)
        self.test.bind(on_press = self.Test)
        self.window.add_widget(self.record)
        self.window.add_widget(self.test)
        return self.window;

    def Record(self, instance):
            print("recording")
    def Test (self, instance):
            print("Testing")
        
if __name__ == "__main__":
    FYPapp().run()