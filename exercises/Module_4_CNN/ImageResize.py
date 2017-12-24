from PIL import Image

img = Image.open('seven.jpg') # image extension *.png,*.jpg
new_width  = 28
new_height = 28
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img2 = img.convert('L')
img2.save('newseven.jpg') # format may what u want ,*.png,*jpg,*.gif

import numpy as np
imgnew = Image.open('newseven.jpg')
im = np.asarray(imgnew)
print(im.shape)