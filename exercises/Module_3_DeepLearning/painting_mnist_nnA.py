import tkinter as tk
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tkinter import messagebox
from PIL import ImageFilter

def NN_minst():
    # # Step 1: Initial Setup
    # X = tf.placeholder(tf.float32, [None, 784])
    # y = tf.placeholder(tf.float32, [None, 10])
    #
    # L1 = 200
    # L2 = 100
    # L3 = 60
    # L4 = 30
    #
    # #[784,L1]  input , number neutron
    # W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
    # B1 = tf.Variable(tf.zeros([L1]))
    # W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
    # B2 = tf.Variable(tf.zeros([L2]))
    # W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
    # B3 = tf.Variable(tf.zeros([L3]))
    # W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
    # B4 = tf.Variable(tf.zeros([L4]))
    # W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
    # B5 = tf.Variable(tf.zeros([10]))
    #
    # # Step 2: Setup Model
    # # Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
    # # Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
    # # Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
    # # Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
    # Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    # Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    # Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    # Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    # Ylogits = tf.matmul(Y4, W5) + B5
    # yhat = tf.nn.softmax(Ylogits)
    #
    # # Step 3: Loss Functions
    # loss = tf.reduce_mean(
    #    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=Ylogits))
    #
    # # Step 4: Optimizer
    # #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.AdamOptimizer()
    # train = optimizer.minimize(loss)
    #
    # # accuracy of the trained model, between 0 (worst) and 1 (best)
    # is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
    # accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

    # sess = tf.Session()
    # saver = tf.train.Saver()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    #
    # # Step 5: Restore
    # saver.restore(sess, "./tmp/mnist.ckpt")
    # print("Model Restore: ")
    #---------------------------------------------------------------

    sess = tf.Session()
    # saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.import_meta_graph('./tmp_a/mnist.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./tmp_a'))
    print("Model Restore: ")
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X_input:0")
    result = graph.get_tensor_by_name("yhat_output:0")
    return sess, result, X


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
        self.nnsess= NN_minst()

    # def save(self):
    #     filename = "temp.jpg"
    #     self.image.save(filename)
    #     self.minst_nn_pred()

    def save(self):
        filename = "temp.jpg"
        self.image.save(filename)
        img = Image.open(filename) # image extension *.png,*.jpg
        new_width  = 28
        new_height = 28
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img2 = img.convert('L')
        img2.save('mnist.jpg') # format may what u want ,*.png,*jpg,*.gif
        self.minst_nn_pred()
        

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

    def minst_nn_pred(self):
        imgnew = Image.open('mnist.jpg')
        im = np.asarray(imgnew)
        print(im.shape)
        im = np.expand_dims(im, axis=0)
        print(im.shape)
        im = im.reshape(1,784)
        print(im.shape)
        classification = self.nnsess[0].run(tf.argmax(self.nnsess[1], 1), feed_dict={self.nnsess[2]: im})
        print('predicted', classification[0])
        messagebox.showinfo("predicted number", str(classification[0]))
        # plt.imshow(im.reshape(28, 28), cmap=plt.cm.binary)
        # plt.show()

    # def minst_nn_pred(self):
    #     imgnew = Image.open('temp.jpg').convert('L')
    #     imgnew = imgnew.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    #     im = np.asarray(imgnew)
    #     im = np.expand_dims(im, axis=0)
    #     im = im.reshape(1,784)
    #     print(im.shape)
    #     classification = self.nnsess[0].run(tf.argmax(self.nnsess[1], 1), feed_dict={self.nnsess[2]: im})
    #     print('predicted', classification[0])
    #     messagebox.showinfo("predicted number", str(classification[0]))
    #     # plt.imshow(im.reshape(28, 28), cmap=plt.cm.binary)
    #     # plt.show()


if __name__ == "__main__":
    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (250, 250, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10)
    root.mainloop()