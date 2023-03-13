import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler



import tensorflow as tf

from keras.layers import Dense, Input, Activation

from keras.layers import BatchNormalization,Add,Dropout

from keras.optimizers import Adam

from keras.models import Model, load_model

from keras import callbacks

from keras import backend as K

from keras.layers.advanced_activations import LeakyReLU

import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action="ignore",category=DeprecationWarning)

warnings.filterwarnings(action="ignore",category=FutureWarning)

import os


print(os.listdir("."))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('train.csv')

df_test=pd.read_csv('test.csv')

df_struct=pd.read_csv('structures.csv')



#df_train_sub_potential=pd.read_csv('/content/champs/potential_energy.csv')

#df_train_sub_moment=pd.read_csv('../input/dipole_moments.csv')

df_train_sub_charge=pd.read_csv('mulliken_charges.csv')

df_train_sub_tensor=pd.read_csv('magnetic_shielding_tensors.csv')

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

print(df_train.shape, df_test.shape, df_struct.shape, df_train_sub_charge.shape, df_train_sub_tensor.shape)

#df_train = reduce_mem_usage(df_train)

#df_test = reduce_mem_usage(df_test)

#df_struct = reduce_mem_usage(df_struct)

#df_train_sub_charge = reduce_mem_usage(df_train_sub_charge)

#df_train_sub_tensor = reduce_mem_usage(df_train_sub_tensor)

print(df_train.shape, df_test.shape, df_struct.shape, df_train_sub_charge.shape, df_train_sub_tensor.shape)
''' 

Map atom info from the structures.csv into the train/test files

'''

import psutil

import os



def map_atom_info(df_1,df_2, atom_idx):

    print('Mapping...', df_1.shape, df_2.shape, atom_idx)

    

    df = pd.merge(df_1, df_2.drop_duplicates(subset=['molecule_name', 'atom_index']), how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)



    return df



def show_ram_usage():

    py = psutil.Process(os.getpid())

    print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))



show_ram_usage()



for atom_idx in [0,1]:

    df_train = map_atom_info(df_train,df_struct, atom_idx)

    df_train = map_atom_info(df_train,df_train_sub_charge, atom_idx)

    df_train = map_atom_info(df_train,df_train_sub_tensor, atom_idx)

    df_train = df_train.rename(columns={'atom': f'atom_{atom_idx}',

                                        'x': f'x_{atom_idx}',

                                        'y': f'y_{atom_idx}',

                                        'z': f'z_{atom_idx}',

                                        'mulliken_charge': f'charge_{atom_idx}',

                                        'XX': f'XX_{atom_idx}',

                                        'YX': f'YX_{atom_idx}',

                                        'ZX': f'ZX_{atom_idx}',

                                        'XY': f'XY_{atom_idx}',

                                        'YY': f'YY_{atom_idx}',

                                        'ZY': f'ZY_{atom_idx}',

                                        'XZ': f'XZ_{atom_idx}',

                                        'YZ': f'YZ_{atom_idx}',

                                        'ZZ': f'ZZ_{atom_idx}',})

    df_test = map_atom_info(df_test,df_struct, atom_idx)

    df_test = df_test.rename(columns={'atom': f'atom_{atom_idx}',

                                'x': f'x_{atom_idx}',

                                'y': f'y_{atom_idx}',

                                'z': f'z_{atom_idx}'})

    #add some features

    

    df_struct['c_x']=df_struct.groupby('molecule_name')['x'].transform('mean')

    df_struct['c_y']=df_struct.groupby('molecule_name')['y'].transform('mean')

    df_struct['c_z']=df_struct.groupby('molecule_name')['z'].transform('mean')

    df_struct['atom_n']=df_struct.groupby('molecule_name')['atom_index'].transform('max')

    

    show_ram_usage()

    print(df_train.shape, df_test.shape)
def make_features(df):

    df['dx']=df['x_1']-df['x_0']

    df['dy']=df['y_1']-df['y_0']

    df['dz']=df['z_1']-df['z_0']

    df['distance']=(df['dx']**2+df['dy']**2+df['dz']**2)**(1/2)

    return df



df_train=make_features(df_train)

df_test=make_features(df_test) 

#df_train = reduce_mem_usage(df_train)

#df_test = reduce_mem_usage(df_test)

test_prediction=np.zeros(len(df_test))

show_ram_usage()

print(df_train.shape, df_test.shape)



def get_dist(df):

    df_temp=df.loc[:,["molecule_name","atom_index_0","atom_index_1","distance","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()

    df_temp_=df_temp.copy()

    df_temp_= df_temp_.rename(columns={'atom_index_0': 'atom_index_1',

                                       'atom_index_1': 'atom_index_0',

                                       'x_0': 'x_1',

                                       'y_0': 'y_1',

                                       'z_0': 'z_1',

                                       'x_1': 'x_0',

                                       'y_1': 'y_0',

                                       'z_1': 'z_0'})

    df_temp_all=pd.concat((df_temp,df_temp_),axis=0)



    df_temp_all["min_distance"]=df_temp_all.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('min')

    df_temp_all["max_distance"]=df_temp_all.groupby(['molecule_name', 'atom_index_0'])['distance'].transform('max')

    

    df_temp= df_temp_all[df_temp_all["min_distance"]==df_temp_all["distance"]].copy()

    df_temp=df_temp.drop(['x_0','y_0','z_0','min_distance'], axis=1)

    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',

                                         'atom_index_1': 'atom_index_closest',

                                         'distance': 'distance_closest',

                                         'x_1': 'x_closest',

                                         'y_1': 'y_closest',

                                         'z_1': 'z_closest'})

    

    for atom_idx in [0,1]:

        df = map_atom_info(df,df_temp, atom_idx)

        df = df.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',

                                        'distance_closest': f'distance_closest_{atom_idx}',

                                        'x_closest': f'x_closest_{atom_idx}',

                                        'y_closest': f'y_closest_{atom_idx}',

                                        'z_closest': f'z_closest_{atom_idx}'})

        

    df_temp= df_temp_all[df_temp_all["max_distance"]==df_temp_all["distance"]].copy()

    df_temp=df_temp.drop(['x_0','y_0','z_0','max_distance'], axis=1)

    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',

                                         'atom_index_1': 'atom_index_farthest',

                                         'distance': 'distance_farthest',

                                         'x_1': 'x_farthest',

                                         'y_1': 'y_farthest',

                                         'z_1': 'z_farthest'})

        

    for atom_idx in [0,1]:

        df = map_atom_info(df,df_temp, atom_idx)

        df = df.rename(columns={'atom_index_farthest': f'atom_index_farthest_{atom_idx}',

                                        'distance_farthest': f'distance_farthest_{atom_idx}',

                                        'x_farthest': f'x_farthest_{atom_idx}',

                                        'y_farthest': f'y_farthest_{atom_idx}',

                                        'z_farthest': f'z_farthest_{atom_idx}'})

    return df

df_test=(get_dist(df_test))    

df_train=(get_dist(df_train)) 



print(df_train.shape, df_test.shape)

show_ram_usage()
def add_features(df):

    df["distance_center0"]=((df['x_0']-df['c_x'])**2+(df['y_0']-df['c_y'])**2+(df['z_0']-df['c_z'])**2)**(1/2)

    df["distance_center1"]=((df['x_1']-df['c_x'])**2+(df['y_1']-df['c_y'])**2+(df['z_1']-df['c_z'])**2)**(1/2)

    df["distance_c0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)

    df["distance_c1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)

    df["distance_f0"]=((df['x_0']-df['x_farthest_0'])**2+(df['y_0']-df['y_farthest_0'])**2+(df['z_0']-df['z_farthest_0'])**2)**(1/2)

    df["distance_f1"]=((df['x_1']-df['x_farthest_1'])**2+(df['y_1']-df['y_farthest_1'])**2+(df['z_1']-df['z_farthest_1'])**2)**(1/2)

    df["vec_center0_x"]=(df['x_0']-df['c_x'])/(df["distance_center0"]+1e-10)

    df["vec_center0_y"]=(df['y_0']-df['c_y'])/(df["distance_center0"]+1e-10)

    df["vec_center0_z"]=(df['z_0']-df['c_z'])/(df["distance_center0"]+1e-10)

    df["vec_center1_x"]=(df['x_1']-df['c_x'])/(df["distance_center1"]+1e-10)

    df["vec_center1_y"]=(df['y_1']-df['c_y'])/(df["distance_center1"]+1e-10)

    df["vec_center1_z"]=(df['z_1']-df['c_z'])/(df["distance_center1"]+1e-10)

    df["vec_c0_x"]=(df['x_0']-df['x_closest_0'])/(df["distance_c0"]+1e-10)

    df["vec_c0_y"]=(df['y_0']-df['y_closest_0'])/(df["distance_c0"]+1e-10)

    df["vec_c0_z"]=(df['z_0']-df['z_closest_0'])/(df["distance_c0"]+1e-10)

    df["vec_c1_x"]=(df['x_1']-df['x_closest_1'])/(df["distance_c1"]+1e-10)

    df["vec_c1_y"]=(df['y_1']-df['y_closest_1'])/(df["distance_c1"]+1e-10)

    df["vec_c1_z"]=(df['z_1']-df['z_closest_1'])/(df["distance_c1"]+1e-10)

    df["vec_f0_x"]=(df['x_0']-df['x_farthest_0'])/(df["distance_f0"]+1e-10)

    df["vec_f0_y"]=(df['y_0']-df['y_farthest_0'])/(df["distance_f0"]+1e-10)

    df["vec_f0_z"]=(df['z_0']-df['z_farthest_0'])/(df["distance_f0"]+1e-10)

    df["vec_f1_x"]=(df['x_1']-df['x_farthest_1'])/(df["distance_f1"]+1e-10)

    df["vec_f1_y"]=(df['y_1']-df['y_farthest_1'])/(df["distance_f1"]+1e-10)

    df["vec_f1_z"]=(df['z_1']-df['z_farthest_1'])/(df["distance_f1"]+1e-10)

    df["vec_x"]=(df['x_1']-df['x_0'])/df["distance"]

    df["vec_y"]=(df['y_1']-df['y_0'])/df["distance"]

    df["vec_z"]=(df['z_1']-df['z_0'])/df["distance"]

    df["cos_c0_c1"]=df["vec_c0_x"]*df["vec_c1_x"]+df["vec_c0_y"]*df["vec_c1_y"]+df["vec_c0_z"]*df["vec_c1_z"]

    df["cos_f0_f1"]=df["vec_f0_x"]*df["vec_f1_x"]+df["vec_f0_y"]*df["vec_f1_y"]+df["vec_f0_z"]*df["vec_f1_z"]

    df["cos_center0_center1"]=df["vec_center0_x"]*df["vec_center1_x"]+df["vec_center0_y"]*df["vec_center1_y"]+df["vec_center0_z"]*df["vec_center1_z"]

    df["cos_c0"]=df["vec_c0_x"]*df["vec_x"]+df["vec_c0_y"]*df["vec_y"]+df["vec_c0_z"]*df["vec_z"]

    df["cos_c1"]=df["vec_c1_x"]*df["vec_x"]+df["vec_c1_y"]*df["vec_y"]+df["vec_c1_z"]*df["vec_z"]

    df["cos_f0"]=df["vec_f0_x"]*df["vec_x"]+df["vec_f0_y"]*df["vec_y"]+df["vec_f0_z"]*df["vec_z"]

    df["cos_f1"]=df["vec_f1_x"]*df["vec_x"]+df["vec_f1_y"]*df["vec_y"]+df["vec_f1_z"]*df["vec_z"]

    df["cos_center0"]=df["vec_center0_x"]*df["vec_x"]+df["vec_center0_y"]*df["vec_y"]+df["vec_center0_z"]*df["vec_z"]

    df["cos_center1"]=df["vec_center1_x"]*df["vec_x"]+df["vec_center1_y"]*df["vec_y"]+df["vec_center1_z"]*df["vec_z"]

    df=df.drop(['vec_c0_x','vec_c0_y','vec_c0_z','vec_c1_x','vec_c1_y','vec_c1_z',

                'vec_f0_x','vec_f0_y','vec_f0_z','vec_f1_x','vec_f1_y','vec_f1_z',

                'vec_center0_x','vec_center0_y','vec_center0_z','vec_center1_x','vec_center1_y','vec_center1_z',

                'vec_x','vec_y','vec_z'], axis=1)

    return df

    

df_train=add_features(df_train)

df_test=add_features(df_test)

print(df_train.shape, df_test.shape)

show_ram_usage()
def create_nn_model(input_shape):

    inp = Input(shape=(input_shape,))

    x = Dense(256)(inp)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.4)(x)

    x = Dense(1024)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.2)(x)

    x = Dense(1024)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.2)(x)

    x = Dense(512)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.4)(x)

    x = Dense(512)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    #x = Dropout(0.4)(x)

    x = Dense(256)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.4)(x)

    out1 = Dense(2, activation="linear")(x)#mulliken charge 2

    out2 = Dense(6, activation="linear")(x)#tensor 6(xx,yy,zz)

    out3 = Dense(12, activation="linear")(x)#tensor 12(others) 

    x = Dense(128)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.2)(x)

    x = Dense(128)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dense(64)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.2)(x)

    out = Dense(1, activation="linear")(x)#scalar_coupling_constant    

    model = Model(inputs=inp, outputs=[out,out1,out2,out3])

    return model
def plot_history(history, label):

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Loss for %s' % label)

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    _= plt.legend(['Train','Validation'], loc='upper left')

    plt.show()
from datetime import datetime



mol_types=df_train["type"].unique()

cv_score=[]

cv_score_total=0

epoch_n = 300

verbose = 0

batch_size = 2048

    

# Set to True if we want to train from scratch.  False will reuse saved models as a starting point.

retrain =False





# Set up GPU preferences

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} ) 

config.gpu_options.allow_growth = True

config.gpu_options.per_process_gpu_memory_fraction = 0.6

sess = tf.Session(config=config) 

K.set_session(sess)



start_time=datetime.now()



# Loop through each molecule type

for mol_type in mol_types:

    model_name_rd = ('../keras-neural-net-for-champs/molecule_model_%s.hdf5' % mol_type)

    model_name_wrt = ('/kaggle/working/molecule_model_%s.hdf5' % mol_type)

    print('Training %s' % mol_type, 'out of', mol_types, '\n')

    

    df_train_=df_train[df_train["type"]==mol_type]

    df_test_=df_test[df_test["type"]==mol_type]

    

    # Here's our best features.  We think.

    input_features=["x_0","y_0","z_0","x_1","y_1","z_1","c_x","c_y","c_z",

                    'x_closest_0','y_closest_0','z_closest_0','x_closest_1','y_closest_1','z_closest_1',

                    "distance","distance_center0","distance_center1", "distance_c0","distance_c1","distance_f0","distance_f1",

                    "cos_c0_c1","cos_f0_f1","cos_center0_center1","cos_c0","cos_c1","cos_f0","cos_f1","cos_center0","cos_center1",

                    "atom_n"

                   ]

    

    # Standard Scaler from sklearn does seem to work better here than other Scalers

    input_data=StandardScaler().fit_transform(pd.concat([df_train_.loc[:,input_features],df_test_.loc[:,input_features]]))

    

    target_data=df_train_.loc[:,"scalar_coupling_constant"].values

    target_data_1=df_train_.loc[:,["charge_0","charge_1"]]

    target_data_2=df_train_.loc[:,["XX_0","YY_0","ZZ_0","XX_1","YY_1","ZZ_1"]]

    target_data_3=df_train_.loc[:,["YX_0","ZX_0","XY_0","ZY_0","XZ_0","YZ_0","YX_1","ZX_1","XY_1","ZY_1","XZ_1","YZ_1"]]

    

    #following parameters should be adjusted to control the loss function

    #if all parameters are zero, attractors do not work. (-> simple neural network)

    m1=1

    m2=4

    m3=1

    target_data_1=m1*(StandardScaler().fit_transform(target_data_1))

    target_data_2=m2*(StandardScaler().fit_transform(target_data_2))

    target_data_3=m3*(StandardScaler().fit_transform(target_data_3))

    

    # Simple split to provide us a validation set to do our CV checks with

    train_index, cv_index = train_test_split(np.arange(len(df_train_)),random_state=111, test_size=0.1)

    

    # Split all our input and targets by train and cv indexes

    train_input=input_data[train_index]

    cv_input=input_data[cv_index]

    train_target=target_data[train_index]

    cv_target=target_data[cv_index]

    train_target_1=target_data_1[train_index]

    cv_target_1=target_data_1[cv_index]

    train_target_2=target_data_2[train_index]

    cv_target_2=target_data_2[cv_index]

    train_target_3=target_data_3[train_index]

    cv_target_3=target_data_3[cv_index]

    test_input=input_data[len(df_train_):,:]



    # Build the Neural Net

    nn_model=create_nn_model(train_input.shape[1])

    

    # If retrain==False, then we load a previous saved model as a starting point.

    if not retrain:

        nn_model = load_model(model_name_rd)

        

    nn_model.compile(loss='mae', optimizer=Adam())#, metrics=[auc])

    

    # Callback for Early Stopping... May want to raise the min_delta for small numbers of epochs

    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=8,verbose=1, mode='auto', restore_best_weights=True)

    # Callback for Reducing the Learning Rate... when the monitor levels out for 'patience' epochs, then the LR is reduced

    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=7, min_lr=1e-6, mode='auto', verbose=1)

    # Save the best value of the model for future use

    sv_mod = callbacks.ModelCheckpoint(model_name_wrt, monitor='val_loss', save_best_only=True, period=1)



    history = nn_model.fit(train_input,[train_target,train_target_1,train_target_2,train_target_3], 

            validation_data=(cv_input,[cv_target,cv_target_1,cv_target_2,cv_target_3]), 

            callbacks=[es, rlr, sv_mod], epochs=epoch_n, batch_size=batch_size, verbose=verbose)

    

    cv_predict=nn_model.predict(cv_input)

    plot_history(history, mol_type)

    

    accuracy=np.mean(np.abs(cv_target-cv_predict[0][:,0]))

    cv_score.append(np.log(accuracy))

    cv_score_total+=np.log(accuracy)

    

    # Predict on the test data set using our trained model

    test_predict=nn_model.predict(test_input)

    

    # for each molecule type we'll grab the predicted values

    test_prediction[df_test["type"]==mol_type]=test_predict[0][:,0]

    K.clear_session()



cv_score_total/=len(mol_types)



def submit(predictions):

    submit = pd.read_csv('sample_submission.csv')

    print(len(submit), len(predictions))   

    submit["scalar_coupling_constant"] = predictions

    submit.to_csv("/kaggle/working/workingsubmission-test.csv", index=False)

submit(test_prediction)



print ('Total training time: ', datetime.now() - start_time)



i=0

for mol_type in mol_types: 

    print(mol_type,": cv score is ",cv_score[i])

    i+=1

print("total cv score is",cv_score_total)
import keras

print(keras.__version__)