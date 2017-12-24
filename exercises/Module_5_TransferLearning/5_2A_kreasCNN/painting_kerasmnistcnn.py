import tkinter as tk
from PIL import Image, ImageDraw
from PIL import ImageFilter
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tkinter import messagebox
import json
from keras.models import model_from_json


def CNN_minst():
    
    ### Load architecture
    with open('cnn_model.json', 'r') as architecture_file:    
        model_architecture = json.load(architecture_file)
    model = model_from_json(model_architecture)
 
    ### Load weights
    model.load_weights('cnn_wt.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    print("Model Restore: ")
    return model


class ImageGenerator:
    def __init__(self,parent,posx,posy,*kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 200
        self.sizey = 200
        self.b1 = "up"
        self.xold = None
        self.yold = None 
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Done!",width=10,bg='white',command=self.save)
        self.button.place(x=self.sizex/7,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Clear!",width=10,bg='white',command=self.clear)
        self.button1.place(x=(self.sizex/7)+80,y=self.sizey+20)

        self.image=Image.new("RGB",(200,200),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)
        self.cnnmodel= CNN_minst()

    # def save(self):
    #     filename = "temp.jpg"
    #     self.image.save(filename)
    #     img = Image.open(filename) # image extension *.png,*.jpg
    #     new_width  = 28
    #     new_height = 28
    #     img = img.resize((new_width, new_height), Image.ANTIALIAS)
    #     img2 = img.convert('L')
    #     img2.save('mnist.jpg') # format may what u want ,*.png,*jpg,*.gif
    #     self.minst_cnn_pred()

    def save(self):
        filename = "temp.jpg"
        self.image.save(filename)
        self.minst_cnn_pred()

    def clear(self):
        self.drawing_area.delete("all")
        self.image=Image.new("RGB",(200,200),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth='true',width=20,fill='black')
                self.draw.line(((self.xold,self.yold),(event.x,event.y)),(0,128,0),width=20)

        self.xold = event.x
        self.yold = event.y

    def minst_cnn_pred(self):
        img = Image.open('temp.jpg').convert('L')
        # resize to 28x28
        img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        # normalization : 255 RGB -> 0, 1
        data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]
        # reshape -> [-1, 28, 28, 1]
        im = np.reshape(data, (-1, 28, 28, 1))
        print(im)
        prediction = self.cnnmodel.predict(im)[0]
        print(prediction)
        bestclass = ''
        bestconf = -1
        for n in [0,1,2,3,4,5,6,7,8,9]:
            if (prediction[n] > bestconf):
                bestclass = str(n)
                bestconf = prediction[n]
        print('I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + ' confidence.')
        messagebox.showinfo("predicted number", bestclass)
        # plt.imshow(im.reshape(28, 28), cmap=plt.cm.binary)
        # plt.show()


if __name__ == "__main__":
    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (250, 250, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10)
    root.mainloop()