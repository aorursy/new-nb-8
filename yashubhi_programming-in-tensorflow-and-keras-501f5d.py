from IPython.display import YouTubeVideo

YouTubeVideo('sDG5tPtsbSA', width=800, height=450)
from os.path import join



image_dir = '../input/dog-breed-identification/train/'

img_paths = [join(image_dir, filename) for filename in 

                    ['0246f44bb123ce3f91c939861eb97fb7.jpg',

                    '84728e78632c0910a69d33f82e62638c.jpg',

                    '8825e914555803f4c67b26593c9d5aff.jpg',

                    '91a5e8db15bccfb6cfa2df5e8b95ec03.jpg']]
import numpy as np

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



image_size = 224



def read_and_prep_images(img_paths, img_height = image_size, img_width = image_size):

    # Loading images in list comprehension

    imgs = [load_img(img_path, target_size = (img_height, img_width)) for img_path in img_paths]

    

    # converting images read in above line to array. Each image is a 3d tensor in RGB format and all images are stacked one over another making this a 4D tensor

    img_array = np.array([img_to_array(img) for img in imgs])

    

    #this normalises the image value to range (-1,1)

    output = preprocess_input(img_array)

    

    return(output)
from tensorflow.python.keras.applications import ResNet50

#this process is similar to scikit-learn where first model is instantiated and then trained. Here we already have trained weights. We directly make predictions by getting the relevant dataset.

my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

test_data = read_and_prep_images(img_paths)

preds = my_model.predict(test_data)
import sys



sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils/')



from decode_predictions import decode_predictions



from IPython.display import Image,display



most_likely_labels = decode_predictions(preds, top=3, class_list_path='../input/resnet50/imagenet_class_index.json')



for i,img_path in enumerate(img_paths):

    display(Image(img_path))

    print(most_likely_labels[i])