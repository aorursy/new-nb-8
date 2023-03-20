import math, re, os

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets  # Cpmment this out when running locally

#import efficientnet.tfkeras as efn

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



print("Tensorflow version " + tf.__version__)



# The Keras library provides support for neural networks and deep learning

from tensorflow import keras

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Lambda, Flatten, LSTM, SpatialDropout2D

from tensorflow.keras.layers import Conv2D, Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam, RMSprop

#from tensorflow.keras.utils import np_utils

from tensorflow.keras import utils

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
# Data access for on Kaggle

GCS_DS_PATH = KaggleDatasets().get_gcs_path()  # you can list the bucket with "!gsutil ls $GCS_DS_PATH"

# Data access for on local machine

#GCS_DS_PATH = "data/"


IMAGE_SIZE = [512, 512] # At this size, a GPU will run out of memory. Use the TPU.

#IMAGE_SIZE = [224, 224] # For GPU training, please select 224 x 224 px image size or you will likely run out of RAM.



EPOCHS = 15           # This is the training time. Can be set long if you use EarlyStopping, but be careful you too much TPU time

#BATCH_SIZE = 32 * strategy.num_replicas_in_sync

#BATCH_SIZE = 32       # Good for a single CPU or GPU



BATCH_SIZE = 128     # 128 is Good for a TPU with multiple processeors



print ("EPOCHS = ", EPOCHS)

print ("BATCH_SIZE = ", BATCH_SIZE)



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition



CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102
# Modify data_augment to change how training images are adjusted during training

# See tf.image for documentation -- https://www.tensorflow.org/api_docs/python/tf/image



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    

    image = tf.image.random_flip_left_right(image)

#     image = tf.image.random_flip_up_down(image)

    image = tf.image.random_brightness(image, 0.1)

    image = tf.image.random_contrast(image, 0.9, 1.0)

#    image = tf.image.random_hue(image, 0.1)

#    image = tf.image.random_contrast(image, 0.1)

    image = tf.image.random_saturation(image, 0.9, 1.0)

    

#     image = tf.image.random_jpeg_quality(image, 85, 100)

    width = IMAGE_SIZE[0]

    large_width = math.floor(width * 1.2)  # increase images sizes by 10% before random crop

    print ("image width = ", width, " resized to ", large_width)

    image = tf.image.resize(image, [large_width, large_width])

    image = tf.image.random_crop(image, [width, width, 3])

    #image = tf.image.random_saturation(image, 0, 2)

    image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=1.)

    return image, label   





# ====== You should not have to edit any functions below here



def data_augment_flip_only(image, label):

    image = tf.image.random_flip_left_right(image)

    return image, label   



AUTO = tf.data.experimental.AUTOTUNE



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



# The class function is used by the balance_dataset to get the image id from each image

def class_func(image, id):

    return id



# Since there are many more images of some flowers than others, this tries to balance the number of each type of flower in the dataset

def balance_dataset(dataset):

    NUM_CLASSES = len(CLASSES)

    PROB = 1 / NUM_CLASSES

    TARGET_DIST = [PROB] * NUM_CLASSES

    print ("balancing the dataset with rejection_resample with distribution = ",TARGET_DIST)

    # TODO --- add code to calculate the number of images in each flower class and generate the initial_dist

    #resampler = tf.data.experimental.rejection_resample(class_func, target_dist=TARGET_DIST, initial_dist=COUNT

    resampler = tf.data.experimental.rejection_resample(class_func, target_dist=TARGET_DIST)

    dataset = dataset.apply(resampler)

    return dataset



def get_training_dataset(augment=False, balance=False):

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    if augment:

        print ("augmenting images in dataset")

        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    else:

        dataset = dataset.map(data_augment_flip_only, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    if balance:

        print("balancing the dataset")

        #dataset = balance_dataset(dataset) 

        NUM_CLASSES = len(CLASSES)

        PROB = 1 / NUM_CLASSES

        TARGET_DIST = [PROB] * NUM_CLASSES

        # TODO --- add code to calculate the number of images in each flower class and generate the initial_dist

        #resampler = tf.data.experimental.rejection_resample(class_func, target_dist=TARGET_DIST, initial_dist=COUNT

        resampler = tf.data.experimental.rejection_resample(class_func, target_dist=TARGET_DIST)

        #dataset = dataset.unbatch()

        resample_ds  = dataset.apply(resampler)

        #dataset = dataset.apply(resampler)

        dataset = resample_ds.map(lambda extra_label, image_and_id: image_and_id)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_train_valid_datasets(augment=False, balance=False):

    dataset = load_dataset(TRAINING_FILENAMES + VALIDATION_FILENAMES, labeled=True)

    if augment:

        print ("augmenting images in dataset")

        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    else:

        dataset = dataset.map(data_augment_flip_only, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    if balance:

        print("balancing the dataset")

        #dataset = balance_dataset(dataset) 

        NUM_CLASSES = len(CLASSES)

        PROB = 1 / NUM_CLASSES

        TARGET_DIST = [PROB] * NUM_CLASSES

        # TODO --- add code to calculate the number of images in each flower class and generate the initial_dist

        #resampler = tf.data.experimental.rejection_resample(class_func, target_dist=TARGET_DIST, initial_dist=COUNT

        resampler = tf.data.experimental.rejection_resample(class_func, target_dist=TARGET_DIST)

        #dataset = dataset.unbatch()

        resample_ds  = dataset.apply(resampler)

        #dataset = dataset.apply(resampler)

        dataset = resample_ds.map(lambda extra_label, image_and_id: image_and_id)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

# numpy and matplotlib defaults

np.set_printoptions(threshold=15, linewidth=80)



def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

        numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

                                CLASSES[correct_label] if not correct else ''), correct



def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)

    

def display_batch_of_images(databatch, predictions=None):

    """This will work with:

    display_batch_of_images(images)

    display_batch_of_images(images, predictions)

    display_batch_of_images((images, labels))

    display_batch_of_images((images, labels), predictions)

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else CLASSES[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()



    

def display_confusion_matrix(cmat, score, precision, recall):

    plt.figure(figsize=(15,15))

    ax = plt.gca()

    ax.matshow(cmat, cmap='Reds')

    ax.set_xticks(range(len(CLASSES)))

    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    ax.set_yticks(range(len(CLASSES)))

    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    titlestring = ""

    if score is not None:

        titlestring += 'f1 = {:.3f} '.format(score)

    if precision is not None:

        titlestring += '\nprecision = {:.3f} '.format(precision)

    if recall is not None:

        titlestring += '\nrecall = {:.3f} '.format(recall)

    if len(titlestring) > 0:

        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})

    plt.show()

    

def display_training_curves(training, validation, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])



# This is a callback to display the learning graphs during training

# I don't know the original author of this method, but is is used frequently in Kaggle competitions and on GitHub

class PlotLearning(Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure() 

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('sparse_categorical_accuracy'))

        self.val_acc.append(logs.get('val_sparse_categorical_accuracy'))

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        #clear_output(wait=True)

        ax1.set_yscale('log')

        ax1.plot(self.x, self.losses, label="loss")

        ax1.plot(self.x, self.val_losses, label="val_loss")

        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")

        ax2.plot(self.x, self.val_acc, label="validation accuracy")

        ax2.legend()

        plt.show();

# data dump

print("Training data shapes:")

for image, label in get_training_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Training data label examples:", label.numpy())

print("Validation data shapes:")

for image, label in get_validation_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Validation data label examples:", label.numpy())

print("Test data shapes:")

for image, idnum in get_test_dataset().take(3):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
# Peek at training data without image augmentation

NUM_IMAGES_TO_DISPLAY = BATCH_SIZE  # can only display images up to the batch size

training_dataset = get_training_dataset()

training_dataset = training_dataset.unbatch().batch(NUM_IMAGES_TO_DISPLAY)

train_batch = iter(training_dataset)
# run this cell again for next set of images

# These images relfect the origina unaugmentated images

print ("images without augmentation")

display_batch_of_images(next(train_batch))

# Peek at training data with image augmentation

NUM_IMAGES_TO_DISPLAY = BATCH_SIZE  # can only display images up to the batch size

training_dataset_augment = get_training_dataset(augment=True)

training_dataset_augment = training_dataset_augment.unbatch().batch(NUM_IMAGES_TO_DISPLAY)

train_batch_augment = iter(training_dataset_augment)
# run this cell again for next set of images

# These images relfect the image augmentation setting above

print ("images with augmentation")

display_batch_of_images(next(train_batch_augment))
# run this cell again for next set of images

display_batch_of_images(next(train_batch))
# run this cell again for next set of images

display_batch_of_images(next(train_batch))
# peer at test data

test_dataset = get_test_dataset()

test_dataset = test_dataset.unbatch().batch(20)

test_batch = iter(test_dataset)
# run this cell again for next set of images

display_batch_of_images(next(test_batch))
with strategy.scope():



    INPUT_SIZE = [*IMAGE_SIZE, 3]

    OUTPUT_SIZE = len(CLASSES)

    print ("INPUT_SIZE = ", INPUT_SIZE)

    print ("OUTPUT_SIZE = ", OUTPUT_SIZE)

    

    #pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=INPUT_SIZE)



    # by default Xception expects images of size 299x299 pixels

    pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=INPUT_SIZE)



    # VGG works with 224x224 size images

    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=INPUT_SIZE)



    pretrained_model.trainable = True      # False = transfer learning, True = fine-tuning

 

    

    model = tf.keras.Sequential([

        pretrained_model,                                 # Include layers in pretrained model from above

        tf.keras.layers.GlobalAveragePooling2D(),

        #tf.keras.layers.Dense(1024, activation="relu"),  # Can add additional layers here

        #tf.keras.layers.Dense(200, activation="relu"),  # Can add additional layers here

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    

# Some sample weight optimizer settings

#RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

optimizer_RMSprop = RMSprop(lr=0.00001, epsilon=1e-08)

#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#optimizer_Adam = Adam(learning_rate=0.001) # default learning rate

optimizer_Adam = Adam(learning_rate=0.0001)

#optimizer_SGD = tf.keras.optimizers.SGD(lr=0.01, 

#                              decay=1e-6, 

#                              momentum=0.9, 

#                              nesterov=True)



model.compile(

#    optimizer=optimizer_Adam,

    optimizer=optimizer_RMSprop,

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)



print ("=== Pretrained Model =========================================================================")

pretrained_model.summary()   # print layers in pretrained model

print ("=== Final Model =========================================================================")

model.summary()              # print final model
# # option to create your own model from scratch



# with strategy.scope():

  

#     model = Sequential()

#     model.add(Conv2D(64, kernel_size=(3,3), activation="relu", input_shape=[*IMAGE_SIZE, 3]))

#     model.add(Conv2D(64, kernel_size=(3,3), activation="relu", padding='same'))

#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))





#     model.add(Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'))

#     model.add(Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'))

#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



#     model.add(Conv2D(256, kernel_size=(3,3), activation="relu", padding='same'))

#     model.add(Conv2D(256, kernel_size=(3,3), activation="relu", padding='same'))

#     model.add(Conv2D(256, kernel_size=(3,3), activation="relu", padding='same'))

#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



#     model.add(Conv2D(512, kernel_size=(3,3), activation="relu", padding='same'))

#     model.add(Conv2D(512, kernel_size=(3,3), activation="relu", padding='same'))

#     model.add(Conv2D(512, kernel_size=(3,3), activation="relu", padding='same'))

#     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



#     #model.add(Conv2D(512, kernel_size=(3,3), activation="relu", padding='same'))

#     #model.add(Conv2D(512, kernel_size=(3,3), activation="relu", padding='same'))

#     #model.add(Conv2D(512, kernel_size=(3,3), activation="relu", padding='same'))

#     #model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



#     model.add(GlobalAveragePooling2D())

#     model.add(Dense(1024, activation="relu"))

#     model.add(Dense(1024, activation="relu"))

#     model.add(Dense(numClasses, activation="softmax"))

        

#     model.compile(

#         optimizer='adam',

#         loss = 'sparse_categorical_crossentropy',

#         metrics=['sparse_categorical_accuracy']

#     )

#     model.summary()
# from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint



# Modify the patience to change how quickly the learning rate is reduced by the factor

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 

                                            patience=2, 

                                            verbose=2, 

                                            factor=0.5,                                            

                                            min_lr=0.0000001)



# Modify the patience to change how quickly the training is stopped once the loss is not dereasing

early_stops = EarlyStopping(monitor='loss', 

                            min_delta=0, 

                            patience=3, 

                            verbose=2, 

                            mode='auto')



# Save the best models

checkpointer = ModelCheckpoint(filepath = 'FowersGPU1.{epoch:02d}-{accuracy:.6f}.hdf5',

                               verbose=2,

                               save_best_only=True, 

                               save_weights_only = True)



# This is the learn rate function used in the original notebook. Works well also

def lrfn(epoch):

    LR_START = 0.00001

    #LR_MAX = 0.00005 * strategy.num_replicas_in_sync

    LR_MAX = 0.00005

    LR_MIN = 0.00001

    LR_RAMPUP_EPOCHS = 5

    LR_SUSTAIN_EPOCHS = 0

    LR_EXP_DECAY = .8

    

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



def freeze(model):

    for layer in model.layers:

        layer.trainable = False



def unfreeze(model):

    for layer in model.layers:

        layer.trainable = True



lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

# Learning rate schedule graph

lrfn_rng = [i for i in range(25 if EPOCHS<25 else EPOCHS)]

lrfn_y = [lrfn(x) for x in lrfn_rng]

a = plt.plot(lrfn_rng, lrfn_y)





       

plot_losses = PlotLearning()
print ("BATCH_SIZE =  ", BATCH_SIZE)

print ("STEPS_PER_EPOCH =  ", STEPS_PER_EPOCH)

print ("EPOCHS =  ", EPOCHS)

print ("initial training W/O image augmentation")



history = model.fit(get_train_valid_datasets(augment=False, balance=False),

#history = model.fit(get_training_dataset(augment=False, balance=False),

                    steps_per_epoch=STEPS_PER_EPOCH, 

                    epochs=EPOCHS, 

                    callbacks=[learning_rate_reduction, early_stops, plot_losses],

                    #callbacks=[early_stopping, reduce_lr, checkpoint, plot_losses],

                    #callbacks=[lr_schedule],

                    validation_data=get_validation_dataset()

                   )
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

display_training_curves(history.history['sparse_categorical_accuracy'], history.history['val_sparse_categorical_accuracy'], 'accuracy', 212)
model.optimizer.lr = 0.000005



print ("STEPS_PER_EPOCH =  ", STEPS_PER_EPOCH)

print ("EPOCHS =  ", EPOCHS)

print ("initial training WITH image augmentation")



history = model.fit(get_train_valid_datasets(augment=True, balance=True), 

#history = model.fit(get_training_dataset(augment=True, balance=True), 

                    steps_per_epoch=STEPS_PER_EPOCH, 

                    epochs=EPOCHS, 

                    callbacks=[learning_rate_reduction, early_stops, plot_losses],

                    #callbacks=[early_stopping, reduce_lr, checkpoint, plot_losses],

                    #callbacks=[lr_schedule],

                    validation_data=get_validation_dataset()

                   )

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))

ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")

ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")

ax_acc.plot(history.epoch, history.history["sparse_categorical_accuracy"], label="Train accuracy")

ax_acc.plot(history.epoch, history.history["val_sparse_categorical_accuracy"], label="Validation accuracy")
cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)
wrongList = []

for correct,predict in zip(cm_correct_labels,cm_predictions):

    if correct != predict:

        #print("Correct flower: ", correct, " Predicted flower: ", predict)

        print(correct,", ", CLASSES[correct],", ", predict, ", ", CLASSES[predict] )

        wrongList.append([correct,predict])

#print (sorted(wrongList))
for img,correct,predict in zip(images_ds, labels_ds, cm_predictions):

    if correct == predict:

        print("Correct flower: ", correct, " Predicted flower: ", predict)
cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

display_confusion_matrix(cmat, score, precision, recall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

dataset = get_validation_dataset()

dataset = dataset.unbatch().batch(50)

batch = iter(dataset)
# run this cell again for next set of images

images, labels = next(batch)

probabilities = model.predict(images)

predictions = np.argmax(probabilities, axis=-1)

display_batch_of_images((images, labels), predictions)