import json



import pandas as pd

import numpy as np

import plotly.express as px

import tensorflow.keras.layers as L

import tensorflow as tf

from sklearn.model_selection import train_test_split
tf.random.set_seed(2020)

np.random.seed(2020)
# This will tell us the columns we are predicting

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
y_true = tf.random.normal((32, 68, 3))

y_pred = tf.random.normal((32, 68, 3))
def MCRMSE(y_true, y_pred):

    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)
def gru_layer(hidden_dim, dropout):

    return L.Bidirectional(L.GRU(

        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))
def build_model(embed_size, seq_len=107, pred_len=68, dropout=0.5, 

                sp_dropout=0.2, embed_dim=200, hidden_dim=256, n_layers=3):

    inputs = L.Input(shape=(seq_len, 3))

    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)

    

    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3])

    )

    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)

    

    for x in range(n_layers):

        hidden = gru_layer(hidden_dim, dropout)(hidden)

    

    # Since we are only making predictions on the first part of each sequence, 

    # we have to truncate it

    truncated = hidden[:, :pred_len]

    out = L.Dense(5, activation='linear')(truncated)

    

    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(tf.optimizers.Adam(), loss=MCRMSE)

    

    return model
def pandas_list_to_array(df):

    """

    Input: dataframe of shape (x, y), containing list of length l

    Return: np.array of shape (x, l, y)

    """

    

    return np.transpose(

        np.array(df.values.tolist()),

        (0, 2, 1)

    )
def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):

    return pandas_list_to_array(

        df[cols].applymap(lambda seq: [token2int[x] for x in seq])

    )
data_dir = '/kaggle/input/stanford-covid-vaccine/'

train = pd.read_json(data_dir + 'train.json', lines=True)

test = pd.read_json(data_dir + 'test.json', lines=True)

sample_df = pd.read_csv(data_dir + 'sample_submission.csv')
train = train.query("signal_to_noise >= 1")
# We will use this dictionary to map each character to an integer

# so that it can be used as an input in keras

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}



train_inputs = preprocess_inputs(train, token2int)

train_labels = pandas_list_to_array(train[pred_cols])
x_train, x_val, y_train, y_val = train_test_split(

    train_inputs, train_labels, test_size=.1, random_state=34, stratify=train.SN_filter)
public_df = test.query("seq_length == 107")

private_df = test.query("seq_length == 130")



public_inputs = preprocess_inputs(public_df, token2int)

private_inputs = preprocess_inputs(private_df, token2int)
model = build_model(embed_size=len(token2int))

model.summary()
history = model.fit(

    x_train, y_train,

    validation_data=(x_val, y_val),

    batch_size=64,

    epochs=75,

    verbose=2,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(patience=5),

        tf.keras.callbacks.ModelCheckpoint('model.h5')

    ]

)
fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'MCRMSE'}, 

    title='Training History')

fig.show()
# Caveat: The prediction format requires the output to be the same length as the input,

# although it's not the case for the training data.

model_public = build_model(seq_len=107, pred_len=107, embed_size=len(token2int))

model_private = build_model(seq_len=130, pred_len=130, embed_size=len(token2int))



model_public.load_weights('model.h5')

model_private.load_weights('model.h5')
public_preds = model_public.predict(public_inputs)

private_preds = model_private.predict(private_inputs)
preds_ls = []



for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)

preds_df.head()
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)