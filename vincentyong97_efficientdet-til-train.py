# import json

# import pandas as pd



# def convert_to_df(json_filepath, filename):

#     with open(json_filepath) as json_file:

#         data = json.load(json_file)

        

    

#     train_annotations = pd.DataFrame(data['annotations'])

#     train_images = pd.DataFrame(data['images'])

#     train_categories = pd.DataFrame(data['categories'])



#     categories_dict = pd.Series(train_categories.name.values,index=train_categories.id).to_dict() # for mapping

#     train_images.columns = ['file_name', 'image_id'] # for merging

    

#     df = pd.merge(train_annotations, train_images, on='image_id', how='left')

    

#     df['file_name'] = df['file_name'].apply(lambda x: '/kaggle/input/til2020/train/train/' + x)

    

#     df['xmin'] = df['bbox'].apply(lambda x: int(x[0]))

#     df['ymin'] = df['bbox'].apply(lambda x: int(x[1]))



#     df['xmax'] = df['bbox'].apply(lambda x: int(x[0] +  x[2]))

#     df['ymax'] = df['bbox'].apply(lambda x: int(x[1] + x[3]))

#     df['class'] = df['category_id'].apply(lambda x: categories_dict[x])

#     df = df.iloc[:, 6:]

#     df.to_csv(filename, index=False)

    

#     return df
# convert_to_df('/kaggle/input/til2020/train.json', 'train_df.csv').head()
# def convert_to_df(json_filepath, filename):

#     with open(json_filepath) as json_file:

#         data = json.load(json_file)

        

    

#     train_annotations = pd.DataFrame(data['annotations'])

#     train_images = pd.DataFrame(data['images'])

#     train_categories = pd.DataFrame(data['categories'])



#     categories_dict = pd.Series(train_categories.name.values,index=train_categories.id).to_dict() # for mapping

#     train_images.columns = ['file_name', 'image_id'] # for merging

    

#     df = pd.merge(train_annotations, train_images, on='image_id', how='left')

    

#     df['file_name'] = df['file_name'].apply(lambda x: '/kaggle/input/til2020/val/val/' + x)

    

#     df['xmin'] = df['bbox'].apply(lambda x: int(x[0]))

#     df['ymin'] = df['bbox'].apply(lambda x: int(x[1]))



#     df['xmax'] = df['bbox'].apply(lambda x: int(x[0] +  x[2]))

#     df['ymax'] = df['bbox'].apply(lambda x: int(x[1] + x[3]))

    

#     df['class'] = df['category_id'].apply(lambda x: categories_dict[x])

#     df = df.iloc[:, 6:]

#     df.to_csv(filename, index=False)

    

#     return df
# convert_to_df('/kaggle/input/til2020/val.json', 'val_df.csv').head()
# data['categories']
# !git clone https://github.com/xuannianz/EfficientDet.git
# import os



# os.chdir('/kaggle/input/efficientdet-til/EfficientDet')
import os

os.chdir('/kaggle/working/EfficientDet')



# !pip install -r requirements.txt
import tensorflow

tensorflow.test.is_gpu_available()

print(tensorflow.__version__)