#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install -U -q kaggle --force ')




from google.colab import files
f=files.upload()




get_ipython().system('mkdir -p ~/.kaggle ')




get_ipython().system('cp kaggle.json ~/.kaggle/')




get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')




get_ipython().system('kaggle competitions download -c nnfl-cnn-lab2')




get_ipython().run_cell_magic('bash', '', 'cd /content\nunzip nnfl-cnn-lab2.zip')




import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("../content"))




FAST_RUN = False
IMAGE_WIDTH=150
IMAGE_HEIGHT=150
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3




# !unzip ../content/train.zip




# !unzip ../content/test1.zip




# filenames = os.listdir("../content/train")
# categories = []
# for filename in filenames:
#     category = filename.split('.')[0]
#     if category == 'dog':
#         categories.append(1)
#     else:
#         categories.append(0)

# df = pd.DataFrame({
#     'filename': filenames,
#     'category': categories
# })
df = pd.read_csv('../content/upload/train_set.csv')




df.head()




df.tail()




df['label'].value_counts().plot.bar()




filenames = os.listdir("../content/upload/train_images/train_images")
sample = random.choice(filenames)
image = load_img("../content/upload/train_images/train_images/"+sample)
plt.imshow(image)




from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, LeakyReLU

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
# model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
# model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(Conv2D(128, (3, 3)))
# model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# (28,28,1) -> (28*28,1) : Flatten
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
# model.add(Dense(64))
# model.add(LeakyReLU(alpha=0.05))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()




from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau




earlystop = EarlyStopping(patience=10)




learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000005)




callbacks = [earlystop, learning_rate_reduction]




df['label'].head()




df["label"] = df["label"].replace({0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}) 




train_df, validate_df = train_test_split(df, test_size=0.20, random_state=41)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)




train_df['label'].value_counts().plot.bar()




validate_df['label'].value_counts().plot.bar()




total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15




print(total_train)
print(total_validate)




train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../content/upload/train_images/train_images/", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)




validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../content/upload/train_images/train_images/", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)




example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "../content/upload/train_images/train_images/", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)




plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()




epochs=3 if FAST_RUN else 50
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)




model.save_weights("model-9.h5")




fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()




# test_filenames = os.listdir("../content/test1/")
# test_df = pd.DataFrame({
#     'filename': test_filenames
# })
# nb_samples = test_df.shape[0]
test_df = pd.read_csv("../content/upload/sample_submission.csv")
# test_df.head(50)




test_df.tail()









test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../content/upload/test_images/test_images/", 
    x_col='image_name',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)




nb_samples = test_df.shape[0]
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))




test_df['label'] = np.argmax(predict, axis=-1)




label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['label'] = test_df['label'].replace(label_map)




test_df['label'] = test_df['label'].replace({ 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5 })




test_df['label'].value_counts().plot.bar()




sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['image_name']
    category = row['label']
    img = load_img("../content/upload/test_images/test_images/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()




submission_df = test_df.copy()
submission_df['image_name'] = submission_df['image_name']
submission_df['label'] = submission_df['label']
#submission_df.drop(['image_name', 'label'], axis=1, inplace=True)
submission_df.head()




submission_df.describe()




submission_df.to_csv('submission9.csv', index=False)

