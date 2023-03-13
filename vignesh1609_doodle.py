# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='white', context='notebook')



np.random.seed(36)
import ast

import cv2

import dask.bag as db



from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

from keras.callbacks import EarlyStopping, ReduceLROnPlateau 



from keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.vgg19 import VGG19
# list of animals 

animals = ['ant', 'bat', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow',

           'crab', 'crocodile', 'dog', 'dolphin', 'dragon', 'duck', 'elephant', 'fish',

           'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse', 'kangaroo', 'lion',

           'lobster', 'monkey', 'mosquito', 'mouse', 'octopus', 'owl', 'panda',

           'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros', 'scorpion',

           'sea turtle', 'shark', 'sheep', 'snail', 'snake', 'spider', 'squirrel',

           'swan', 'teddy-bear', 'tiger', 'whale', 'zebra']
dir_path='/kaggle/input/quickdraw-doodle-recognition/train_simplified/'

df = pd.read_csv(dir_path + animals[0] + '.csv')

df.head()
am = pd.DataFrame(columns = df.columns)



for i in range(len(animals)):

    filename = dir_path + animals[i] + '.csv'

    df = pd.read_csv(filename, nrows = 100)

    df = df[df.recognized == True]

    am = am.append(df)
am.head()
am.word.nunique()
# Sampling only 100 examples

ex = am.sample(100)

# Convert to list as drawing columns is in strings

ex['drawing'] = ex.drawing.map(ast.literal_eval)
fig, axs = plt.subplots(nrows = 10, ncols = 10, figsize = (10, 8))



for index, col in enumerate(ex.drawing):

    ax = axs[index//10, index%10]

    for x, y in col:

        ax.plot(x,-np.array(y), lw = 3)

    ax.axis('off')

    

plt.show()
im_size = 64

n_class = len(animals)



# redefine

def draw_to_img(strokes, im_size = im_size):

    fig, ax = plt.subplots()                        # plot the drawing as we did above

    for x, y in strokes:

        ax.plot(x, -np.array(y), lw = 10)

    ax.axis('off')

    

    fig.canvas.draw()                               # update a figure that has been altered

    A = np.array(fig.canvas.renderer._renderer)     # converting them into array

    

    plt.close('all')

    plt.clf()

    

    A = (cv2.resize(A, (im_size, im_size)) / 255.)  # image resizing to uniform format



    return A[:, :, :3]        
X = ex.drawing.values

image = draw_to_img(X[1])

plt.imshow(image)
image.shape
im_size = 64

n_class = len(animals)

n_samples = 500

X_train = np.zeros((1, im_size, im_size, 3))

y = []



for a in animals:

    #print(a)

    filename = dir_path + a + '.csv'

    df = pd.read_csv(filename, usecols=['drawing', 'word'], nrows=n_samples)  # import the data in chunks

    df['drawing'] = df.drawing.map(ast.literal_eval)                          # convert strings into list

    X = df.drawing.values

    

    img_bag = db.from_sequence(X).map(draw_to_img)                            # covert strokes into array

    X = np.array(img_bag.compute())  

    X_train = np.vstack((X_train, X))                                         # concatenate to get X_train  

    

    y.append(df.word)
X_train.shape
# Drop the first layer

X_train = X_train[1:, :, :, :]

X_train.shape
#Encoding 

y = pd.DataFrame(y)

y = pd.get_dummies(y)

y_train = np.array(y).transpose()
#Training data

print("The input shape is {}".format(X_train.shape))

print("The output shape is {}".format(y_train.shape))
X_train[0][1][2]
# Reshape X_train

X_train_2 = X_train.reshape((X_train.shape[0], im_size*im_size*3))



# Concatenate X_train and y_train

X_y_train = np.hstack((X_train_2, y_train))
# Random shuffle

np.random.shuffle(X_y_train)

a = im_size*im_size*3

cut = int(len(X_y_train) * .1)

X_val = X_y_train[:cut, :a]

y_val = X_y_train[:cut, a:]

X_train = X_y_train[cut:, :a]

y_train = X_y_train[cut:, a:]



# Reshape X_train back to (64, 64)

X_train = X_train.reshape((X_train.shape[0], im_size, im_size, 3))

X_val = X_val.reshape((X_val.shape[0], im_size, im_size, 3))
#final Shape

print("The input shape of train set is {}".format(X_train.shape))

print("The input shape of validation set is {}".format(X_val.shape))

print("The output shape of train set is {}".format(y_train.shape))

print("The output shape of validation set is {}".format(y_val.shape))
n_epochs = 10

batch_size = 500



# Initialize

model = Sequential()



# ConvNet_1

model.add(Conv2D(32, kernel_size = 3, input_shape = (im_size, im_size, 3), padding = 'same', activation = 'relu'))

model.add(MaxPool2D(2, strides = 2))

# Dropout

model.add(Dropout(.2))



# ConvNet_2

model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))

model.add(MaxPool2D(2, strides = 2))

# Dropout

model.add(Dropout(.2))



# ConvNet_3

model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))

model.add(MaxPool2D(2, strides = 2))

# Dropout

model.add(Dropout(.2))



# Flattening

model.add(Flatten())



# Fully connected

model.add(Dense(680, activation = 'relu'))



# Dropout

model.add(Dropout(.5))



# Final layer

model.add(Dense(n_class, activation = 'softmax'))



# Compile

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
# Fitting baseline

history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size, 

                    validation_split = .2, verbose = True)
# ResNet50 Application 

model_r = ResNet50(include_top = True, weights= None, input_shape=(im_size, im_size, 3), classes = n_class)
model_r.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model_r.summary()
n_epochs = 5

batch_size = 50
# Fitting ResNet50

history_r = model_r.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size, 

                      validation_split = .2, verbose = True)
# Train and validation curves with ResNet50

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(history_r.history['loss'], color = 'b', label = 'Train Loss')

ax1.plot(history_r.history['val_loss'], color = 'm', label = 'Valid Loss')

ax1.legend(loc = 'best')



ax2.plot(history_r.history['acc'], color = 'b', label = 'Train Accuracy')

ax2.plot(history_r.history['val_acc'], color = 'm', label = 'Valid Accuracy')

ax2.legend(loc = 'best')