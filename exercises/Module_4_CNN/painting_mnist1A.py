import tkinter as tk
from PIL import Image, ImageDraw
from PIL import ImageFilter
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tkinter import messagebox

def CNN_minst():
    # Step 1: Initial Setup
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, 10])
    pkeep = tf.placeholder(tf.float32)

    L1 = 4  # first convolutional filters
    L2 = 8  # second convolutional filters
    L3 = 16  # third convolutional filters
    L4 = 256  # fully connected neurons

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, L1], stddev=0.1))
    B1 = tf.Variable(tf.zeros([L1]))
    W2 = tf.Variable(tf.truncated_normal([3, 3, L1, L2], stddev=0.1))
    B2 = tf.Variable(tf.zeros([L2]))
    W3 = tf.Variable(tf.truncated_normal([3, 3, L2, L3], stddev=0.1))
    B3 = tf.Variable(tf.zeros([L3]))
    W4 = tf.Variable(tf.truncated_normal([7 * 7 * L3, L4], stddev=0.1))
    B4 = tf.Variable(tf.zeros([L4]))
    W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
    B5 = tf.Variable(tf.zeros([10]))

    # Step 2: Setup Model
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)  # output is 28x28
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
    Y2 = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # output is 14x14
    Y2 = tf.nn.dropout(Y2, 0.5)
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, 1, 1, 1], padding='SAME') + B3)
    Y3 = tf.nn.max_pool(Y3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # output is 7x7
    Y3 = tf.nn.dropout(Y3, 0.5)

    # Flatten the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * L3])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    YY4 = tf.nn.dropout(Y4, 0.5)
    Ylogits = tf.matmul(Y4, W5) + B5
    yhat = tf.nn.softmax(Ylogits)

    # Step 3: Loss Functions
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))

    # Step 4: Optimizer
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    # Step 5: Restore
    saver.restore(sess, "./tmp_a/mnist.ckpt")
    print("Model Restore: ")
    return sess, yhat ,X


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
        self.cnnsess= CNN_minst()

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
        im = np.reshape(data, (-1, 28, 28, 1)).tolist()
        print(im)
        classification = self.cnnsess[0].run(tf.argmax(self.cnnsess[1], 1), feed_dict={self.cnnsess[2]: im})
        print('predicted', classification[0])
        messagebox.showinfo("predicted number", str(classification[0]))
        # plt.imshow(im.reshape(28, 28), cmap=plt.cm.binary)
        # plt.show()


if __name__ == "__main__":
    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (250, 250, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10)
    root.mainloop()