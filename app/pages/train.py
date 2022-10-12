import time
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow import keras
from tensorflow.keras import layers
from css.cssHelper import local_css, remote_css, icon

st.set_page_config(page_icon = '🦋')

local_css("./app/css/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

dumps = r'./app/dumps/'

df = pd.read_csv(dumps + 'dataset.csv')
# print(df.head())

st.write("Training data")
st.dataframe(df, 800, 300)

x = np.array(df['x'])
sin_x = np.array(df['sin(x)'])

x = x.reshape((x.shape[0],1))
sin_x = sin_x.reshape((sin_x.shape[0],1))

activation = st.text_input('Activation', 'relu')
lrate = st.select_slider('Learning Rate', [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0])
earlyStopping = st.checkbox("Early Stopping", value = True)

model = keras.Sequential([
        layers.Dense(1, activation = activation, input_shape= [x.shape[1]])
    ])

# model.summary()   # Summarise the model parameters and layers
duhOptimizer = keras.optimizers.Adam(learning_rate = lrate)
# Let’s compile. 
model.compile(optimizer = duhOptimizer, loss = 'mse', metrics = ['mse', 'mae'])

cbacks = []
if earlyStopping:
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    cbacks.append(earlyStopping)

train_start = st.button('Train using the sample data')

if train_start:
    with st.spinner(text='Training in progress'):  
        train_history = model.fit(x, sin_x, epochs = 100, verbose = 1, validation_split = 0.1, callbacks = cbacks)
        st.info("Training has been completed")

        # dump training history information
        hist = pd.DataFrame(train_history.history)
        hist.to_csv(dumps + 'train_history.csv')