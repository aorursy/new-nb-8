#https://www.kaggle.com/anokas/quick-data-analysis



import numpy as np

import pandas as pd

#import matplotlib.pylot as plt



df_train = pd.read_csv('../input/train_masks.csv')

print('Shape of training data:' + str(df_train.shape) + '\n')

print(df_train.head())
pixels = df_train['pixels'].values

print(pixels[0])

p1 = [] # run-start pixel locations

p2=[] # run-lengths

p3=[] #number of data points per image



#separate run-lengths and pixel locations into separate lists

for p in pixels:

    x = str(p).split(' ')

    i=0

    for m in x:

        if i%2 ==0:

            p1.append(m)

        else:

            p2.append(m)

        i +=1

        

# Get number of data points in each image

i=0

for p in pixels:

    x = str(p).split(' ')

    if len(x) == 1:

        p3.append(0)

    else:

        p3.append(len(x)/2)

    i +=1

    

# Get all absolute target values

targets = []

for start,length in zip(p1,p2):

    i =0

    length = int(length)

    if start != 'nan':

        pix = int(start)

        while i <=length:

            targets.append(pix)

            pix +=1

            i+=1

            

print('\nTotal number of target pixels:' + str(len(targets)))



#Remove NaNs

p4 = []

i=0

for p in p1:

    if p == 'nan':

        i +=1

    else:

        p4.append(p)

p1 = p4

print('\nNumber of NaN in pixel locations: ' + str(i))

        

        
print('Number of pixel locations: ' + str(len(p1)))

print('    Number of run lengths: ' + str(len(p2)))

print('\nAverage number of pixel locations per image: ' + str(len(p1) / len(df_train.index)))
import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

from matplotlib.colors import LogNorm



p = np.array(p2).astype(int)

plt.hist(p, 25, normed=1, facecolor='red', alpha=0.75)

plt.title('Histogram of run-lengths')

plt.xlabel('Run length')

plt.ylabel('Frequency')

plt.grid(True)

plt.show()



p = np.array(p1).astype(int)

plt.hist(p, 50, normed=1, facecolor='blue', alpha=0.75)

plt.title('Histogram of pixel location')

plt.xlabel('Pixel location')

plt.ylabel('Frequency')

plt.grid(True)

plt.show()



p = np.array(p3).astype(int)

plt.hist(p, 50, normed=1, facecolor='green', alpha=0.75)

plt.title('Histogram of data point count')

plt.xlabel('Number of data points in image')

plt.ylabel('Frequency')

plt.grid(True)

plt.show()



print("Now let's remove the images with zero target pixels and check again")



px = []

for x in p.tolist():

    if x != 0:

        px.append(x)

        

p = np.array(px).astype(int)

plt.hist(p, 50, normed=1, facecolor='green', alpha=0.75)

plt.title('Histogram of data point count')

plt.xlabel('Number of data points in image')

plt.ylabel('Frequency')

plt.grid(True)

plt.show()
# Display the first 20 images



import matplotlib.pyplot as plt

import glob, os



ultrasounds = [img for img in glob.glob("../input/train/*.tif") if 'mask' not in img]

    

for file in ultrasounds[0:20]:

    im = plt.imread(file)

    plt.figure(figsize=(15,20))

    plt.imshow(im, cmap="Greys_r")

    plt.show()
import glob, os, cv2

ultrasounds = [img for img in glob.glob("../input/train/*.tif") if 'mask' not in img]



img = cv2.imread(ultrasounds[0])

height, width, channels = img.shape

print('Image dimensions: ' + str(height) + 'h x ' + str(width) + 'w - ' + str(channels) + ' channels')




xs = []

ys = []



for p in targets:

    p = int(p)

    xs.append(p % 580)

    ys.append(int(p / 580)) # int() helpfully rounds down

    

bins = 40

while bins <=320:

    plt.hist2d(xs, ys, bins=bins, norm=LogNorm())

    plt.colorbar()

    plt.title('Target pixel location histogram - ' + str(bins) + ' bins')

    plt.xlabel('x')

    plt.ylabel('y')

    plt.show()

    bins = bins * 2


